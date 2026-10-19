# CLAUDE.md

A learning project: a CSV-analysis AI agent built several ways (plain loop, MCP, LangGraph), with agentic
patterns added one stage at a time (code execution, reflection, planning, structured output + guardrails,
human-in-the-loop, observability, multi-agent, long-term memory/RAG, autonomy, web app). The owner is learning
how agents work, so clarity beats cleverness. Map of stages to files: README.md.

## Layout

- `apps/` entry points, run as modules from the root: `python -m apps.<name>`. No logic beyond wiring + CLI.
- `src/core/` base agent and tools · `src/frameworks/` MCP + LangGraph versions · `src/patterns/` one module
  per stage · `src/web/` FastAPI app + static page · `src/paths.py` locations of generated files.
- `runtime/` holds everything generated (SQLite DBs, charts, reports, traces). Git-ignored; new generated
  files go there via `src/paths.py`, never in the repo root.
- `tests/` one `test_<module>.py` per module; `tests/fakes.py` has both fake models.

## Commands

- Python lives in `venv/` (not `.venv/`): use `./venv/Scripts/python.exe` (Windows).
- Tests: `./venv/Scripts/python.exe -m pytest -q`. They use scripted fake models (`FakeClient` for the Groq
  client, `FakeChat` for LangChain, both in `tests/fakes.py`) and need no API key. Run them after every change.
- Eval: `./venv/Scripts/python.exe -m eval.run_eval --versions <v...> --only <case ids...>`.
  This calls the real Groq API on the user's free-tier key: **always run a small subset** (`--only`, few
  `--versions`), never `--versions all` on all cases unless asked. Write scratch runs with `--out` to a temp
  path; `eval/results.md` is the user's.
- The `apps.chat*` modules are interactive (`input()`) and can't be driven non-interactively. Exceptions:
  `apps.report --yes`, and `apps.web` (a server on 127.0.0.1:8000; start it in the background, drive it with
  curl against `/api/...`, stop it when done). Web tests use FastAPI's TestClient.
- Screenshots: headless Edge works, but its minimum viewport is 496 px; narrower `--window-size` values
  just crop the image, so test phone layouts at 496.

## Conventions

- Keep each stage in its own small module with a docstring that explains the concept and its trade-offs.
  Match the existing style: short functions, few comments, explain *why*, plain Python before frameworks.
- New behaviour gets a test with the fake models. Anything seen failing live gets a regression test with a
  comment saying so (e.g. Groq `tool_use_failed`, `output_parse_failed`, `json_validate_failed`).
- Tool results are returned to the model as data, including errors (`{"error": ...}`); never crash the loop
  on a tool problem.
- Every tool result passes through `guardrails.sanitize` (in `CsvTools._out`). Keep it that way for new tools.
- Model-written code only runs through `src/core/sandbox.py`; `run_pandas` is in `hitl.RISKY_TOOLS`.
- The tool descriptions exist in three places (`src/core/schemas.py`, `src/frameworks/mcp_server.py`,
  `langgraph_agent.make_tools`). This duplication is deliberate (it's one of the comparisons): change all three.
- After a live eval, read the failures before trusting the score. Many FAILs have been harness or grader
  bugs. Record real findings in `docs/COMPARISON.md` with the date.

## Gotchas

- `mcp` is version 2 (`MCPServer`, snake_case attributes); v1 tutorials (`FastMCP`) don't apply.
- Model: `openai/gpt-oss-120b` on Groq (`src/core/agent.py: MODEL`). Prices in `src/patterns/tracing.py: PRICES`
  are placeholders.
- Git and `venv/`: the first commits tracked `venv/`; a later commit untracked it. Checking out or merging
  across that commit DELETES the venv files from disk (happened 2026-10-19). Fix with
  `./venv/Scripts/python.exe -m pip install --force-reinstall -r requirements.txt`. Warn the user before
  suggesting git operations that cross it.
- Bash heredocs containing apostrophes fail in this environment's shell tool; write files with the editor tools.
