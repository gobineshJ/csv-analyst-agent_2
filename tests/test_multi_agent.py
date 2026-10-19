import json
import os
from pathlib import Path

from src.core.tools import CsvTools
from src.patterns.multi_agent import run_multi_agent
from tests.fakes import FakeClient, call, text

CSV = os.path.abspath("data/sample_sales.csv")


def plan(*tasks, answer=""):
    return text(json.dumps({"tasks": [{"worker": w, "task": t} for w, t in tasks], "answer": answer}))


def test_single_analyst_task_skips_the_synthesizer():
    client = FakeClient([plan(("analyst", "Compute total sales")), call("aggregate", '{"column":"sales","how":"sum"}'),
                         text("Total sales are 5540.")])
    answer, reports = run_multi_agent("total?", CsvTools(CSV), client=client, verbose=False)
    assert answer == "Total sales are 5540." and len(client.requests) == 3


def test_orchestrator_delegates_to_isolated_specialists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # charts are saved under ./runtime/outputs
    client = FakeClient([
        plan(("analyst", "Compute total sales by category"), ("chart", "Bar chart of sales by category")),
        call("aggregate", '{"column":"sales","how":"sum","group_by":"category"}'), text("Electronics 2760"),
        call("make_chart", '{"kind":"bar","x":"category","y":"sales"}'), text("Saved runtime/outputs/sum_of_sales_by_category.png"),
        text("Electronics leads with 2760. Chart: runtime/outputs/sum_of_sales_by_category.png"),
    ])
    answer, reports = run_multi_agent("Which category leads? Show a chart.", CsvTools(CSV), client=client, verbose=False)
    assert "2760" in answer and [r["worker"] for r in reports] == ["analyst", "chart"]
    assert Path("runtime/outputs/sum_of_sales_by_category.png").exists()
    # The chart specialist starts fresh: it sees its own task, not the analyst's work or the user's question.
    chart_first = client.requests[3]
    assert [m["content"] for m in chart_first if m["role"] == "user"] == ["Bar chart of sales by category"]
    # ...and only has its own tools.
    assert {t["function"]["name"] for t in client.kwargs[3]["tools"]} == {"describe_data", "make_chart"}
    assert "Electronics 2760" in client.requests[5][-1]["content"]  # synthesizer sees the reports


def test_out_of_scope_question_is_answered_without_specialists():
    client = FakeClient([plan(answer="The data has no cost column, so profit can't be computed.")])
    answer, reports = run_multi_agent("profit?", CsvTools(CSV), client=client, verbose=False)
    assert "no cost column" in answer and reports == []


def test_broken_orchestrator_falls_back_to_one_analyst():
    client = FakeClient([text("nope")] * 3 + [text("5540")])
    answer, reports = run_multi_agent("total?", CsvTools(CSV), client=client, verbose=False)
    assert answer == "5540" and reports[0]["task"] == "total?"
