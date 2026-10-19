# CLAUDE.md

A learning project: a CSV-analysis AI agent built several ways (plain loop, MCP, LangGraph), with agentic
patterns added one stage at a time (code execution, reflection, planning, structured output + guardrails,
human-in-the-loop, observability, multi-agent, long-term memory/RAG, autonomy). The owner is learning how
agents work, so clarity beats cleverness. Map of stages to files: README.md.

## Commands

- Python lives in `venv/` (not `.venv/`): use `./venv/Scripts/python.exe` (Windows).
- Tests: `./venv/Scripts/python.exe -m pytest -q`. They use scripted fake models (`tests/fakes.py` for the Groq
  client, `tests/fake_llm.py` for LangChain) and need no API key. Run them after every change.
- Eval: `./venv/Scripts/python.exe -m eval.run_eval --versions <v...> --only <case ids...>`.
  This calls the real Groq API on the user's free-tier key: **always run a small subset** (`--only`, few
  `--versions`), never `--versions all` on all cases unless asked. Write scratch runs with `--out` to a temp
  path; `eval/results.md` is the user's.
- Each `main*.py` is an interactive chat (`input()`); they can't be driven non-interactively, except
  `main_report.py --yes`.

## Conventions

- Keep each stage in its own small module with a docstring that explains the concept and its trade-offs.
  Match the existing style: short functions, few comments, explain *why*, plain Python before frameworks.
- New behaviour gets a test with the fake models. Anything seen failing live gets a regression test with a
  comment saying so (e.g. Groq `tool_use_failed`, `json_validate_failed`).
- Tool results are returned to the model as data, including errors (`{"error": ...}`); never crash the loop
  on a tool problem.
- Every tool result passes through `guardrails.sanitize` (in `CsvTools._out`). Keep it that way for new tools.
- Model-written code only runs through `src/sandbox.py`; `run_pandas` is in `hitl.RISKY_TOOLS`.
- The tool descriptions exist in three places (`schemas.py`, `mcp_server.py`, `langgraph_agent.make_tools`).
  This duplication is deliberate (it's one of the comparisons): change all three together.
- After a live eval, read the failures before trusting the score. Many FAILs have been harness or grader
  bugs. Record real findings in `docs/COMPARISON.md` with the date.

## Gotchas

- `mcp` is version 2 (`MCPServer`, snake_case attributes); v1 tutorials (`FastMCP`) don't apply.
- Model: `openai/gpt-oss-120b` on Groq (`src/agent.py: MODEL`). Prices in `src/tracing.py: PRICES` are
  placeholders.
- Generated files (`outputs/`, `traces/`, `*.db`, `eval/results.json`) are git-ignored.
- `venv/` was committed in the first commit; it's in `.gitignore` now but still tracked until
  `git rm -r --cached venv` is run (ask the user first).
- Bash heredocs containing apostrophes fail in this environment's shell tool; write files with the editor tools.
