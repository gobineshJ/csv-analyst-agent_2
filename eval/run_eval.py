"""Run the same questions through all three agent versions and grade them against pandas.

    python -m eval.run_eval                       # all versions, all cases
    python -m eval.run_eval --versions plain mcp  # only some versions
    python -m eval.run_eval --only total_sales    # only some cases

Writes a markdown table (default eval/results.md) and the raw records as JSON next to it.
Grading is a heuristic (numbers/words found in the answer, refusal keywords), so read the
answers too: a PASS can still contain a wrong extra claim, and a FAIL can be a format issue.
"""
import argparse
import asyncio
import contextlib
import io
import json
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
    turn_ids: list = field(default_factory=list)


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
    Case("misspelled_column", ["What's the average of the 'unit' column?"],
         lambda df: df.units.mean(), kind="misspelling"),
    Case("since_date", ["What were the total sales on or after 2025-01-10?"],
         lambda df: df[df.date >= "2025-01-10"].sales.sum(), kind="known-limitation"),
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
                     r"only (?:have|contain)|beyond|outside", re.I)


def numbers_in(text):
    out = []
    for n in NUMBER.findall(text):
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


def grade(expected, answer):
    """Return (passed, expected_as_text)."""
    if expected == REFUSE:
        return bool(REFUSAL.search(answer)), "refuse / say tools can't answer"
    parts = expected if isinstance(expected, list) else [expected]
    parts = [p.item() if hasattr(p, "item") else p for p in parts]  # numpy -> python
    shown = ", ".join(f"{p:g}" if isinstance(p, float) else str(p) for p in parts)
    return all(part_found(p, answer) for p in parts), shown


# --- trace parsing (all three versions print CALL / RESULT lines when verbose) ----------

def parse_trace(stdout):
    calls = [line.split()[1] for line in stdout.splitlines() if line.startswith("CALL ")]
    errors = sum(1 for line in stdout.splitlines()
                 if line.startswith("RESULT") and re.search(r"""["']error["']""", line))
    return calls, errors


# --- runners: each yields one record per case ------------------------------------------

def record(case, version, answer, stdout, seconds, note=""):
    calls, errors = parse_trace(stdout)
    return {"case": case.id, "kind": case.kind, "version": version, "question": case.turns[-1],
            "answer": answer, "calls": calls, "tool_errors": errors,
            "seconds": round(seconds, 1), "note": note, "trace": stdout}


def run_plain(cases, path, delay, client=None):
    from src.agent import llm_summarizer, run_agent
    from src.memory import ConversationMemory
    from src.tools import CsvTools

    tools = CsvTools(path)
    for case in cases:
        memory = ConversationMemory(summarize=llm_summarizer(client), max_tokens=case.memory_budget)
        buf, start, answer, note = io.StringIO(), time.time(), "", ""
        try:
            with contextlib.redirect_stdout(buf):
                for q in case.turns:
                    answer = run_agent(q, tools, memory=memory, client=client)
            note = "summarized" if memory.summary else ""
        except Exception as e:
            answer = f"ERROR: {type(e).__name__}: {e}"
        yield record(case, "plain", answer, buf.getvalue(), time.time() - start, note)
        time.sleep(delay)


async def _run_mcp(cases, path, delay):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from src.agent import llm_summarizer
    from src.agent_mcp import run_agent_mcp, to_openai_tools
    from src.memory import ConversationMemory

    out = []
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server", path])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = to_openai_tools((await session.list_tools()).tools)
            for case in cases:
                memory = ConversationMemory(summarize=llm_summarizer(), max_tokens=case.memory_budget)
                buf, start, answer, note = io.StringIO(), time.time(), "", ""
                try:
                    with contextlib.redirect_stdout(buf):
                        for q in case.turns:
                            answer = await run_agent_mcp(q, session, tools, memory=memory)
                    note = "summarized" if memory.summary else ""
                except Exception as e:
                    answer = f"ERROR: {type(e).__name__}: {e}"
                out.append(record(case, "mcp", answer, buf.getvalue(), time.time() - start, note))
                await asyncio.sleep(delay)
    return out


def run_mcp(cases, path, delay):
    yield from asyncio.run(_run_mcp(cases, path, delay))


def run_langgraph(cases, path, delay, llm=None):
    from langchain_core.messages.utils import count_tokens_approximately

    from src.langgraph_agent import build_graph, run_agent_langgraph
    from src.tools import CsvTools

    csv = CsvTools(path)
    for case in cases:
        buf, start, answer, note = io.StringIO(), time.time(), "", ""
        try:
            graph = build_graph(csv, llm=llm, max_context_tokens=case.memory_budget)
            with contextlib.redirect_stdout(buf):
                for q in case.turns:
                    answer = run_agent_langgraph(q, graph, thread_id=case.id)
            history = graph.get_state({"configurable": {"thread_id": case.id}}).values["messages"]
            note = "trimmed" if count_tokens_approximately(history) > case.memory_budget else ""
        except Exception as e:
            answer = f"ERROR: {type(e).__name__}: {e}"
        yield record(case, "langgraph", answer, buf.getvalue(), time.time() - start, note)
        time.sleep(delay)


RUNNERS = {"plain": run_plain, "mcp": run_mcp, "langgraph": run_langgraph}

# --- report ----------------------------------------------------------------------------

def cell(text, limit=300):
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return text.replace("|", "\\|")


def to_markdown(records, path, model):
    lines = ["# Evaluation results", "",
             f"Generated by `python -m eval.run_eval` on {time.strftime('%Y-%m-%d %H:%M')}, "
             f"model `{model}`, data `{path}`.",
             "Correct answers are computed with pandas from the same CSV. Grading is automatic and "
             "heuristic: read the answers before trusting a PASS or FAIL.", "",
             "## Summary", "",
             "| Version | Passed | Avg tool calls | Tool errors | Avg seconds |",
             "|---------|--------|----------------|-------------|-------------|"]
    for v in RUNNERS:
        rs = [r for r in records if r["version"] == v]
        if not rs:
            continue
        passed = sum(r["passed"] for r in rs)
        lines.append(f"| {v} | {passed}/{len(rs)} | {sum(len(r['calls']) for r in rs) / len(rs):.1f} "
                     f"| {sum(r['tool_errors'] for r in rs)} | {sum(r['seconds'] for r in rs) / len(rs):.1f} |")
    lines += ["", "## Results", "",
              "| Question | Kind | Version | Agent answer | Correct (pandas) | Match? | Notes |",
              "|----------|------|---------|--------------|------------------|--------|-------|"]
    for r in records:
        notes = [f"calls: {' → '.join(r['calls']) or 'none'}"]
        if r["tool_errors"]:
            notes.append(f"{r['tool_errors']} tool error(s)")
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
    parser.add_argument("--versions", nargs="+", choices=list(RUNNERS), default=list(RUNNERS))
    parser.add_argument("--only", nargs="+", metavar="CASE_ID", help="run only these case ids")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds to wait between cases (free-tier rate limits)")
    parser.add_argument("--out", default="eval/results.md")
    args = parser.parse_args(argv)

    cases = [c for c in CASES if not args.only or c.id in args.only]
    if not cases:
        parser.error(f"no cases match {args.only}. Ids: {[c.id for c in CASES]}")
    df = pd.read_csv(args.data)

    records = []
    for version in args.versions:
        print(f"Running {version} on {len(cases)} case(s)...", file=sys.stderr)
        by_id = {c.id: c for c in cases}
        for r in RUNNERS[version](cases, args.data, args.delay):
            case = by_id[r["case"]]
            r["passed"], r["expected"] = grade(case.expected if case.expected == REFUSE
                                               else case.expected(df), r["answer"])
            print(f"  {version:9} {r['case']}: {'PASS' if r['passed'] else 'FAIL'}", file=sys.stderr)
            records.append(r)

    from src.agent import MODEL
    out = Path(args.out)
    out.write_text(to_markdown(records, args.data, MODEL), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out} and {out.with_suffix('.json')}", file=sys.stderr)


if __name__ == "__main__":
    main()
