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

## To fill in after running the evaluation

- Accuracy of each version on the same questions:
- Which one recovered best from a wrong tool call:
- Did summarizing change any number the user later asked about:
- Which one was easiest to debug from its printed trace:
