# CSV Analyst Agent

A small AI agent that answers plain-English questions about a CSV file by calling pandas-based tools. It is built three ways so the approaches can be compared:

1. **Plain loop** (`main.py`): the agent loop written by hand, no framework.
2. **MCP** (`main_mcp.py`): the same tools served by an MCP server, with the agent discovering them as an MCP client.
3. **LangGraph** (`main_langgraph.py`): the same agent rebuilt as a LangGraph graph.

The first two share a conversation memory that summarizes old turns. LangGraph uses its own checkpointer and message trimming.

## How the agent works

1. The question goes to the LLM together with descriptions of the available tools.
2. If the model wants data, it replies with a tool call instead of an answer.
3. The program runs the tool and sends the result back to the model.
4. Steps 2 and 3 repeat until the model answers, or a step limit is reached.

```
question -> LLM -> tool call? --yes--> run tool -> result back to LLM
                      |
                      no
                      v
                  final answer
```

## Project structure

```
csv-analyst-agent/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── .env.example
├── main.py                 # plain loop, command-line chat
├── main_mcp.py             # MCP client agent, command-line chat
├── main_langgraph.py       # LangGraph agent, command-line chat
├── data/
│   └── sample_sales.csv
├── src/
│   ├── tools.py            # pandas functions, no AI
│   ├── schemas.py          # tool descriptions for the plain loop
│   ├── memory.py           # conversation memory (trim + summarize)
│   ├── agent.py            # plain agent loop
│   ├── mcp_server.py       # the same tools as an MCP server
│   ├── agent_mcp.py        # agent loop that uses the MCP server
│   └── langgraph_agent.py  # LangGraph version
├── tests/                  # run without an API key
├── eval/
│   └── results.md          # question / answer / correct-answer table
└── docs/
    └── COMPARISON.md       # plain loop vs MCP vs LangGraph
```

## Tools

| Tool | What it does |
|------|--------------|
| `describe_data` | Column names, types, row count and sample rows |
| `aggregate` | sum, mean, count, max or min of a column, optionally grouped by another column |
| `filter_rows` | Rows matching a condition (`==`, `!=`, `>`, `<`, `>=`, `<=`, `contains`) |
| `top_n` | Top N rows sorted by a column |

- The model cannot run arbitrary code. It can only call these four fixed functions.
- Tool errors (unknown column, wrong type) are returned to the model as data so it can correct itself.
- Results are capped at 20 rows to keep the model's context small.

## Memory

`src/memory.py` keeps the last few turns word for word and folds older turns into a short summary that is added to the system prompt.

- Compaction starts when the history passes about 3000 estimated tokens (`max_tokens`), and the last 2 turns (`keep_turns`) are always kept.
- Only whole turns are dropped, so a tool call is never separated from its result (the API rejects that).
- The summary is written by the model. If that call fails, a fallback keeps each old question and its final answer word for word.
- Risk: a model-written summary can change a number. The prompt asks it to keep numbers exactly, but this is not guaranteed.
- A single question that needs a very long chain of tool calls is not trimmed while it is running.

LangGraph instead keeps the full history in a checkpointer and shows the model a trimmed view (`trim_messages`). Old turns are dropped, not summarized.

## MCP

`src/mcp_server.py` exposes the four tools over stdio. `main_mcp.py` starts that server as a subprocess, asks it for its tool list, and runs the same agent loop with `session.call_tool()`. The tool descriptions come from the server's docstrings and type hints, so `schemas.py` is not used in this mode.

The server can also be used by any other MCP client:

```bash
python -m src.mcp_server data/sample_sales.csv
```

This project uses `mcp` version 2 (`MCPServer`). Tutorials for version 1 use `FastMCP` and will not run on it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Get a free key at [console.groq.com](https://console.groq.com) and put it in `.env` as `GROQ_API_KEY`. Never commit `.env`.

## Run

```bash
python main.py                      # plain loop
python main_mcp.py                  # via MCP
python main_langgraph.py            # LangGraph
python main.py path/to/your.csv     # any of them with your own file
```

All three print each tool call and result. Try a follow-up that depends on an earlier answer ("and for the South region?") to see memory working.

## Tests

```bash
pytest
```

The tests use a scripted fake model, so they need no API key. They cover the tools, memory, the agent loop, a real MCP server connection over stdio, and the LangGraph graph. They check the plumbing only. They do not show how well a real model chooses tools.

## Evaluation

`eval/run_eval.py` runs a fixed set of questions through all three versions, computes the correct answer for each with pandas from the same CSV, and grades the agent's answer automatically.

```bash
python -m eval.run_eval                         # all versions, all cases (needs GROQ_API_KEY)
python -m eval.run_eval --versions plain mcp    # only some versions
python -m eval.run_eval --only total_sales predict
python -m eval.run_eval --delay 5               # slower, for free-tier rate limits
```

- It writes [`eval/results.md`](eval/results.md), which is overwritten on each run, and `eval/results.json` with the full CALL/RESULT trace of every answer.
- The cases cover plain lookups, a multi-step question, a misspelled column, a date filter (a known limitation), unanswerable questions, and a follow-up asked after memory has been compacted (`memory_budget=400` forces summarizing in plain/MCP and trimming in LangGraph).
- Grading is a heuristic. A number passes if it appears in the answer within 0.5%. Text passes if the word appears. A refusal passes if refusal wording appears. Read the answers before trusting a PASS or FAIL.
- To add a case, append a `Case(...)` to `CASES` with a pandas function for the correct answer.

The comparison of the three approaches is in [`docs/COMPARISON.md`](docs/COMPARISON.md).

## Known limitations

- Free-tier models are less reliable at tool calling, so wrong arguments or tool choices can happen.
- The automated tests use a fake model. Behaviour with a live model is only covered by the evaluation table.
- Token counts are estimated at about 4 characters per token, not measured.
- `filter_rows` takes the value as a string, so date comparisons are not supported properly.
- Nothing prevents the model from misreading a correct tool result. Only comparing against pandas catches this.
- Groq's free tier has rate limits, and available models change.
- Tested only on the small sample dataset.
- In LangGraph, the memory is kept in RAM and is lost when the program exits.

## Planned

- Persist memory across runs
- Date-aware filtering
