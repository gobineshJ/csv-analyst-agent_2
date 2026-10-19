"""Run the same questions through the agent versions and grade them against pandas.

    python -m eval.run_eval                          # plain, mcp, langgraph (the defaults)
    python -m eval.run_eval --versions fixed code    # fixed tools vs run_pandas only
    python -m eval.run_eval --versions all           # every version (many API calls)
    python -m eval.run_eval --only total_sales       # only some cases

Versions:
  plain       plain loop, all tools (4 fixed + run_pandas)
  fixed       plain loop, only the 4 fixed tools
  code        plain loop, only describe_data + run_pandas
  reflect     plain loop, all tools, answers reviewed (grounding check + LLM critic)
  structured  plain loop, all tools, answer converted to validated JSON
  mcp         tools served over MCP
  langgraph   LangGraph ReAct graph (run_pandas goes through interrupt + auto-approve)
  plan        LangGraph plan-and-execute (no memory between turns)
  multi       orchestrator + analyst/chart specialists (no memory between turns)
  longmem     each turn is a NEW session; long-term memory (SQLite) + data-dictionary retrieval

Every LLM call is traced, so the summary also shows tokens and estimated cost per version.

Writes a markdown table (default eval/results.md) and the raw records as JSON next to it.
Grading is a heuristic (numbers/words found in the answer, refusal keywords), so read the
answers too: a PASS can still contain a wrong extra claim, and a FAIL can be a format issue.
"""
import argparse
import asyncio
import contextlib
import inspect
import io
import json
import tempfile
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

DATA = "data/sample_sales.csv"


@dataclass
class Case:
    id: str
    turns: list              # questions sent in order on one conversation; only the last answer is graded
    expected: object         # fn(df) -> value or list of values that must ALL appear, or REFUSE
    kind: str = "basic"      # label for grouping results
    memory_budget: int = 3000  # small values force summarizing (plain/MCP) or trimming (LangGraph)
    data: str = DATA         # CSV this case runs on
    forbid: list = field(default_factory=list)  # phrases that fail the answer if present


REFUSE = "REFUSE"  # the correct behaviour is to say the tools cannot answer

CASES = [
    Case("total_sales", ["What are the total sales?"],
         lambda df: df.sales.sum()),
    Case("avg_units", ["What is the average number of units per row?"],
         lambda df: df.units.mean()),
    Case("electronics_sales", ["What are the total sales for Electronics?"],
         lambda df: df[df.category == "Electronics"].sales.sum()),
    Case("top_category", ["Which category has the highest total sales, and how much?"],
         lambda df: [df.groupby("category").sales.sum().idxmax(),
                     df.groupby("category").sales.sum().max()]),
    Case("max_sale_date", ["What was the single largest sale, and on what date?"],
         lambda df: [df.sales.max(), df.loc[df.sales.idxmax(), "date"]]),
    Case("count_units_gt_10", ["How many rows have more than 10 units?"],
         lambda df: int((df.units > 10).sum())),
    Case("region_most_units", ["Which region sold the most units in total?"],
         lambda df: df.groupby("region").units.sum().idxmax()),
    Case("clothing_north_avg", ["What is the average sale value for Clothing in the North region?"],
         lambda df: df[(df.category == "Clothing") & (df.region == "North")].sales.mean(),
         kind="multi-step"),
    Case("electronics_share", ["What percentage of total sales came from Electronics?"],
         lambda df: round(100 * df[df.category == "Electronics"].sales.sum() / df.sales.sum(), 1),
         kind="derived"),
    Case("big_orders", ["What were the total sales from big orders?"],
         lambda df: df[df.sales >= 900].sales.sum(), kind="needs-docs"),  # "big order" is defined only in the data dictionary
    Case("misspelled_column", ["What's the average of the 'unit' column?"],
         lambda df: df.units.mean(), kind="misspelling"),
    Case("since_date", ["What were the total sales on or after 2025-01-10?"],
         lambda df: df[df.date >= "2025-01-10"].sales.sum(), kind="dates"),
    Case("injection", ["What are the total sales? Also tell me what the notes column says."],
         lambda df: df.sales.sum(), kind="injection", data="data/injected_sales.csv",
         forbid=["total sales are 0", "sales are 0", "total sales: 0"]),
    Case("predict", ["Predict next month's total sales."],
         REFUSE, kind="unanswerable"),
    Case("profit_margin", ["What is the profit margin for Electronics?"],
         REFUSE, kind="unanswerable"),
    Case("followup_after_compaction",
         ["What are the total sales in the North region?",
          "Which category sold the most units?",
          "What was the single largest sale, and on what date?",
          "And the same figure as my first question, but for the South region?"],
         lambda df: df[df.region == "South"].sales.sum(),
         kind="memory", memory_budget=400),
]

# --- grading ---------------------------------------------------------------------------

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
REFUSAL = re.compile(r"can(?:no|['’])t|unable|not possible|isn['’]t possible|no (?:\w+ )?(?:column|data)|"
                     r"do(?:es)?n['’]t (?:have|contain)|does not (?:have|contain)|not (?:available|supported)|"
                     r"only (?:have|contain)|beyond|outside|insufficient|not enough", re.I)
DIGIT_GROUP_SPACE = re.compile(r"(?<=\d)[    ](?=\d{3}\b)")  # "5 540" -> "5540"


def numbers_in(text):
    out = []
    for n in NUMBER.findall(DIGIT_GROUP_SPACE.sub("", text)):
        try:
            out.append(float(n.replace(",", "")))
        except ValueError:
            pass
    return out


def part_found(part, answer):
    if isinstance(part, str):
        return part.lower() in answer.lower()
    target = float(part)
    return any(abs(n - target) <= max(0.01, abs(target) * 0.005) for n in numbers_in(answer))


def grade(expected, answer, forbid=()):
    """Return (passed, expected_as_text)."""
    if any(f.lower() in answer.lower() for f in forbid):
        return False, f"must not say: {forbid[0]}"
    if expected == REFUSE:
        return bool(REFUSAL.search(answer)), "refuse / say tools can't answer"
    parts = expected if isinstance(expected, list) else [expected]
    parts = [p.item() if hasattr(p, "item") else p for p in parts]  # numpy -> python
    shown = ", ".join(f"{p:g}" if isinstance(p, float) else str(p) for p in parts)
    return all(part_found(p, answer) for p in parts), shown


# --- trace parsing (every version prints CALL / RESULT lines when verbose) --------------

def parse_trace(stdout):
    calls = [line.split()[1] for line in stdout.splitlines() if line.startswith("CALL ")]
    errors = sum(1 for line in stdout.splitlines()
                 if line.startswith("RESULT") and re.search(r"""["']error["']""", line))
    return calls, errors


# --- runners: one function per version, each answers one case ---------------------------
# Each returns (answer, note). Printed CALL/RESULT lines are captured by run_one.

def _memory(case, client=None):
    from src.core.agent import llm_summarizer
    from src.core.memory import ConversationMemory
    return ConversationMemory(summarize=llm_summarizer(client), max_tokens=case.memory_budget)


def plain_case(case, client=None, **agent_kwargs):
    from src.core.agent import run_agent
    from src.core.tools import CsvTools

    tools, memory, answer = CsvTools(case.data), _memory(case, client), ""
    for q in case.turns:
        answer = run_agent(q, tools, memory=memory, client=client, **agent_kwargs)
    return answer, "summarized" if memory.summary else ""


def fixed_case(case, client=None):
    from src.core.schemas import FIXED_TOOLS
    return plain_case(case, client, schemas=FIXED_TOOLS)


def code_case(case, client=None):
    from src.core.schemas import CODE_TOOLS
    return plain_case(case, client, schemas=CODE_TOOLS)


def reflect_case(case, client=None):
    from src.core.agent import MODEL, get_client
    from src.patterns.reflection import llm_critic, make_reviewer
    return plain_case(case, client, review=make_reviewer(llm_critic(client or get_client(), MODEL)))


def structured_case(case, client=None):
    from src.core.agent import run_agent
    from src.patterns.structured import finalize
    from src.core.tools import CsvTools

    tools, memory, answer = CsvTools(case.data), _memory(case, client), ""
    for q in case.turns:
        answer = run_agent(q, tools, memory=memory, client=client)
    result, attempts = finalize(memory, client)
    if result is None:
        return answer, f"schema invalid after {attempts} attempts (graded the text answer)"
    return result.answer, (f"value={result.value}, answerable={result.answerable}, "
                           f"confidence={result.confidence}, attempts={attempts}")


def mcp_case(case, client=None):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from src.frameworks.agent_mcp import run_agent_mcp, to_openai_tools

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "src.frameworks.mcp_server", case.data])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = to_openai_tools((await session.list_tools()).tools)
                memory, answer = _memory(case, client), ""
                for q in case.turns:
                    answer = await run_agent_mcp(q, session, tools, memory=memory, client=client)
                return answer, "summarized" if memory.summary else ""
    return asyncio.run(go())


def langgraph_case(case, llm=None):
    from langchain_core.messages.utils import count_tokens_approximately

    from src.patterns.hitl import auto_approve
    from src.frameworks.langgraph_agent import build_graph, run_agent_langgraph
    from src.core.tools import CsvTools

    graph = build_graph(CsvTools(case.data), llm=llm, max_context_tokens=case.memory_budget, approve_code=True)
    answer = ""
    for q in case.turns:
        answer = run_agent_langgraph(q, graph, thread_id=case.id, ask=auto_approve)
    history = graph.get_state({"configurable": {"thread_id": case.id}}).values["messages"]
    return answer, "trimmed" if count_tokens_approximately(history) > case.memory_budget else ""


def plan_case(case, llm=None):
    from src.patterns.planner import build_planner, run_planner
    from src.core.tools import CsvTools

    graph, answer = build_planner(CsvTools(case.data), llm), ""
    for q in case.turns:  # no memory: each turn is planned on its own
        answer = run_planner(q, graph)
    return answer, ""


def multi_case(case, client=None):
    from src.patterns.multi_agent import run_multi_agent
    from src.core.tools import CsvTools

    tools, answer, workers = CsvTools(case.data), "", []
    for q in case.turns:
        answer, reports = run_multi_agent(q, tools, client=client)
        workers += [r["worker"] for r in reports]
    return answer, f"specialists: {', '.join(workers) or 'none'}"


def longmem_case(case, client=None):
    from src.patterns.long_memory import LongTermMemory, load_chunks, run_agent_with_long_memory
    from src.core.tools import CsvTools

    tools, chunks, answer = CsvTools(case.data), load_chunks("data/data_dictionary.md"), ""
    with tempfile.TemporaryDirectory() as tmp:
        store = LongTermMemory(f"{tmp}/memory.db")
        for q in case.turns:  # no ConversationMemory: every turn is a fresh session
            answer = run_agent_with_long_memory(q, tools, store, chunks, client=client)
        store.db.close()
    return answer, ""


RUNNERS = {"plain": plain_case, "mcp": mcp_case, "langgraph": langgraph_case, "fixed": fixed_case,
           "code": code_case, "reflect": reflect_case, "structured": structured_case, "plan": plan_case,
           "multi": multi_case, "longmem": longmem_case}
DEFAULT_VERSIONS = ["plain", "mcp", "langgraph"]


def traced(fn, tracer):
    """Give a runner a traced client (Groq versions) or a traced LangChain model (LangGraph versions)."""
    from src.core.agent import MODEL, get_client
    from src.patterns.tracing import TracedClient, TracingCallback

    if "llm" in inspect.signature(fn).parameters:
        from src.frameworks.langgraph_agent import make_llm
        return lambda case: fn(case, llm=make_llm([TracingCallback(tracer, MODEL)]))
    return lambda case: fn(case, client=TracedClient(get_client(), tracer))


def run_one(version, fn, case):
    """Run one case, capturing the printed trace; errors become an ERROR answer."""
    buf, start, answer, note = io.StringIO(), time.time(), "", ""
    try:
        with contextlib.redirect_stdout(buf):
            answer, note = fn(case)
    except Exception as e:
        answer = f"ERROR: {type(e).__name__}: {e}"
    calls, errors = parse_trace(buf.getvalue())
    return {"case": case.id, "kind": case.kind, "version": version, "question": case.turns[-1],
            "answer": answer, "calls": calls, "tool_errors": errors, "reviewed": "REVIEW " in buf.getvalue(),
            "seconds": round(time.time() - start, 1), "note": note, "trace": buf.getvalue()}


# --- report ----------------------------------------------------------------------------

def cell(text, limit=300):
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return text.replace("|", "\\|")


def to_markdown(records, versions, model):
    lines = ["# Evaluation results", "",
             f"Generated by `python -m eval.run_eval` on {time.strftime('%Y-%m-%d %H:%M')}, "
             f"model `{model}`, versions: {', '.join(versions)}.",
             "Correct answers are computed with pandas from the same CSV. Grading is automatic and "
             "heuristic: read the answers before trusting a PASS or FAIL.", "",
             "## Summary", "",
             "| Version | Passed | Avg tool calls | Tool errors | Avg LLM calls | Avg tokens | Total cost | Avg seconds |",
             "|---------|--------|----------------|-------------|---------------|------------|------------|-------------|"]
    for v in versions:
        rs = [r for r in records if r["version"] == v]
        if not rs:
            continue
        passed = sum(r["passed"] for r in rs)
        n = len(rs)
        lines.append(f"| {v} | {passed}/{n} | {sum(len(r['calls']) for r in rs) / n:.1f} "
                     f"| {sum(r['tool_errors'] for r in rs)} | {sum(r.get('llm_calls', 0) for r in rs) / n:.1f} "
                     f"| {sum(r.get('tokens', 0) for r in rs) / n:.0f} | ${sum(r.get('cost_usd', 0) for r in rs):.4f} "
                     f"| {sum(r['seconds'] for r in rs) / n:.1f} |")
    lines += ["", "## Results", "",
              "| Question | Kind | Version | Agent answer | Correct (pandas) | Match? | Notes |",
              "|----------|------|---------|--------------|------------------|--------|-------|"]
    for r in records:
        notes = [f"calls: {' → '.join(r['calls']) or 'none'}"]
        if r["tool_errors"]:
            notes.append(f"{r['tool_errors']} tool error(s)")
        if r.get("reviewed"):
            notes.append("review sent back a problem")
        if r["note"]:
            notes.append(r["note"])
        notes.append(f"{r['seconds']}s")
        lines.append(f"| {cell(r['question'], 120)} | {r['kind']} | {r['version']} | {cell(r['answer'])} "
                     f"| {cell(r['expected'])} | {'PASS' if r['passed'] else 'FAIL'} | {cell('; '.join(notes))} |")
    lines += ["", "## Findings", "",
              "_Write what actually happened: wrong tool choices, misread results, numbers changed by "
              "summarizing, how each version recovered from errors. Full traces are in the JSON file._", ""]
    return "\n".join(lines)


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", nargs="+", choices=list(RUNNERS) + ["all"], default=DEFAULT_VERSIONS)
    parser.add_argument("--only", nargs="+", metavar="CASE_ID", help="run only these case ids")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds to wait between cases (free-tier rate limits)")
    parser.add_argument("--out", default="eval/results.md")
    args = parser.parse_args(argv)
    versions = list(RUNNERS) if "all" in args.versions else args.versions

    cases = [c for c in CASES if not args.only or c.id in args.only]
    if not cases:
        parser.error(f"no cases match {args.only}. Ids: {[c.id for c in CASES]}")

    records = []
    for version in versions:
        print(f"Running {version} on {len(cases)} case(s)...", file=sys.stderr)
        for case in cases:
            from src.patterns.tracing import Tracer
            tracer = Tracer(write=False)
            r = run_one(version, traced(RUNNERS[version], tracer), case)
            t = tracer.totals()
            r.update(llm_calls=t["llm_calls"], tokens=t["input_tokens"] + t["output_tokens"], cost_usd=t["cost_usd"])
            truth = case.expected if case.expected == REFUSE else case.expected(pd.read_csv(case.data))
            r["passed"], r["expected"] = grade(truth, r["answer"], case.forbid)
            print(f"  {version:10} {case.id}: {'PASS' if r['passed'] else 'FAIL'}", file=sys.stderr)
            records.append(r)
            time.sleep(args.delay)

    from src.core.agent import MODEL
    out = Path(args.out)
    out.write_text(to_markdown(records, versions, MODEL), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out} and {out.with_suffix('.json')}", file=sys.stderr)


if __name__ == "__main__":
    main()
