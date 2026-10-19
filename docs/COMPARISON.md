# Plain loop vs MCP vs LangGraph

What is written here comes from reading and testing the code with a scripted fake model.
It says nothing about how the three behave with a live model. Fill in the last section after running the evaluation.

## Size (measured with `wc -l`, non-blank non-comment lines)

| File | Lines | What it contains |
|------|-------|------------------|
| `src/agent.py` | 64 | loop, tool execution, client setup, LLM summarizer |
| `src/agent_mcp.py` | 45 | loop, tool execution through MCP |
| `src/langgraph_agent.py` | 70 | tool definitions, graph, run helper |
| `src/schemas.py` | 77 | tool descriptions for the plain loop |
| `src/mcp_server.py` | 51 | the same four tools as an MCP server |
| `src/memory.py` | 66 | memory, used by the plain and MCP versions |

LangGraph is not shorter here. Its 70 lines include about 30 lines of tool definitions, but the loop itself and
message storage are handled by the framework.

## Where each concern lives

| Concern | Plain loop | MCP | LangGraph |
|---------|-----------|-----|-----------|
| Loop | `for` loop in `agent.py` | same loop, `async` | graph edges (`model` <-> `tools`) |
| Tool descriptions | `schemas.py` (hand-written) | docstrings and type hints in the server | docstrings and type hints on `@tool` |
| Running a tool | `getattr(tools, name)(**args)` | `session.call_tool(name, args)` | `ToolNode` |
| Unknown tool / bad arguments | my `ALLOWED` check and `try/except` | the server rejects it and returns error text | `ToolNode` returns an error message |
| Step limit | `max_steps` counter | `max_steps` counter | `recursion_limit` setting (raises an exception) |
| Memory across questions | `ConversationMemory` | `ConversationMemory` | checkpointer + `thread_id` |
| Long conversations | summarize old turns | summarize old turns | `trim_messages` (old turns are dropped, not summarized) |
| Provider message format | handled by hand (`tool_calls`, `tool_call_id`) | handled by hand | hidden by `ChatGroq` |

## Things found while building, and what they imply

1. **Tool descriptions are written three times** (`schemas.py`, the MCP server, the `@tool` functions). With MCP the
   client needs no copy at all: it asks the server. That is the practical gain of MCP here.
2. **MCP moves the allow-list to the server.** The client no longer decides which tools exist.
3. **LangGraph's step limit leaves a broken thread.** When `recursion_limit` is hit, the saved history ends with a tool
   call that has no result. Sending that on the next question is invalid. The plain loop does not have this problem
   because it always finishes the tool calls of a step before checking the limit. The fix is in `run_agent_langgraph`
   and is covered by a test.
4. **Memory approaches differ in what they lose.** Summarizing keeps the gist of old turns but can change a number.
   Trimming loses old turns completely but never alters what is kept. Which is better depends on whether follow-up
   questions refer back to old results. Neither was tested with a live model.
5. **Dependencies move quickly.** `mcp` 2.x renamed `FastMCP` to `MCPServer` and changed attribute names to snake_case.
   Code written for 1.x will not run on 2.x, so `requirements.txt` pins `mcp>=2`.

## Stages 1–5: first live run (2026-10-16)

One run of 5 hard cases (multi-step, percentage, dates, injection, predict) through all 8 versions with
`openai/gpt-oss-120b` on Groq. A single run is noisy, so treat this as hypotheses to re-test, not results.

| Version | Passed | Avg tool calls | Avg seconds |
|---------|--------|----------------|-------------|
| code (describe_data + run_pandas) | 5/5 | 2.0 | 10.7 |
| plain (all 5 tools) | 4/5 | 2.4 | 7.1 |
| structured | 4/5 | 3.6 | 29.0 |
| plan | 4/5 | 2.8 | 32.0 |
| mcp | 3/5 | 3.6 | 31.6 |
| langgraph | 3/5 | 4.0 | 28.4 |
| reflect | 3/5 | 3.6 | 20.7 |
| fixed (4 fixed tools only) | 2/5 | 3.0 | 19.0 |

What it showed:

1. **Code execution is the biggest single gain.** `fixed` failed the multi-step and date questions, which the
   4 tools can't express. `code` solved them in fewer calls than any other version.
2. **More capability creates new failures.** With `run_pandas`, three versions *predicted* next month's sales
   (answers ranged from 5,540 to 20,775) instead of refusing. The fixed tools couldn't forecast, so they
   refused. A tool that can do anything makes the system prompt's "say so if the tools can't answer"
   much weaker. A re-run after the error-handling fixes gave the same result (19,390 twice, 20,775 once).
   Reflection did not catch it either: the number really did come from a `run_pandas` result, so the
   grounding check passed. Grounding checks where a number came from, not whether the question should have
   been answered at all. Fix to try: a sharper prompt, or a critic rule about forecasts.
3. **Providers can reject tool calls before your code sees them.** Groq returned `400 tool_use_failed` when the
   model wrote `"arguments": result = ...` (code not wrapped in a JSON string). This crashed the plain, MCP and
   LangGraph runs. Code is the argument type most likely to break JSON. Fixed by retrying once with a hint
   (`agent.chat()`), like any other tool error.
4. **Reflection is not free and not automatically better.** On the percentage question, the review sent the
   answer back, and the model then spent its remaining steps re-aggregating until it hit the step limit.
   Reflection needs a step budget of its own.
5. **Every extra LLM call is a new failure point.** The critic's JSON-mode call itself failed
   (`json_validate_failed`). It now fails open: a broken reviewer lets the answer through rather than crashing.
6. **The guardrail held in every version.** No answer repeated "total sales are 0". (The one injection FAIL was
   a grader bug: "5 540" written with a narrow no-break space.)
7. **Plan-and-execute was the slowest** (≈3× plain), with similar accuracy on these questions.
8. **The grader needed fixing twice**, for number formatting and the refusal wording "insufficient data".
   Read failures before trusting them. 6 of the 12 FAILs in this run came from missing error handling or
   grader bugs, not from the model reasoning badly.

## Stages 7–8: plain vs multi-agent vs long-term memory (2026-10-19)

One run of 4 cases with `openai/gpt-oss-120b` on Groq. Tokens and cost come from the stage-6 tracer.

| Version | Passed | Avg LLM calls | Avg tokens | Avg seconds |
|---------|--------|---------------|------------|-------------|
| plain | 2/4 | 4.8 | 3,307 | 10.4 |
| multi | 2/4 | 5.8 | 3,512 | 24.7 |
| longmem | 4/4 | 5.0 | 3,629 | 26.9 |

1. **Retrieval supplies knowledge the data doesn't have.** `big_orders` passed only in `longmem`, which retrieved
   the data-dictionary definition (sales ≥ 900). `plain` and `multi` correctly refused rather than guessing a
   threshold. Refusing was right; being able to answer was better.
2. **Memory between sessions works where the session summary used to fail.** `longmem` ran every turn as a
   brand-new session and still answered "the same figure as my first question" (1,600), because recall always
   includes the most recent notes, not only keyword matches.
3. **Multi-agent cost more and gained nothing here.** It averaged 2.4× the time of `plain`, with the same
   accuracy. On the memory question it failed twice over: the specialists have no conversation memory, and
   the orchestrator read "figure" as a chart. Splitting work pays off when tasks are big and different, not
   for short single-number questions.
4. **A third provider error appeared.** Groq returned `output_parse_failed` (unparseable model output) on the
   percentage question, which crashed `plain`. It is now retried like `tool_use_failed`, with a regression test.
   Each new model behaviour seen live tends to need its own handling, which is why live evals matter.

## To fill in after running the evaluation

- Accuracy of each version on the same questions:
- Which one recovered best from a wrong tool call:
- Did summarizing change any number the user later asked about:
- Which one was easiest to debug from its printed trace:
