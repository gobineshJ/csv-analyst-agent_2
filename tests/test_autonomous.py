import json
import os

from src.core.tools import CsvTools
from src.patterns.autonomous import run_autonomous
from src.patterns.tracing import Budget
from tests.fakes import FakeClient, call, text

CSV = os.path.abspath("data/sample_sales.csv")
GOAL = "Short sales report"
TASKS = [{"id": "t1", "worker": "analyst", "task": "Total sales"},
         {"id": "t2", "worker": "chart", "task": "Bar chart of sales by category"},
         {"id": "t3", "worker": "analyst", "task": "Average units by category"}]
PLAN = text(json.dumps({"tasks": TASKS}))
T1 = [call("aggregate", '{"column":"sales","how":"sum"}', "a"), text("Total sales are 5540.")]
T2 = [call("make_chart", '{"kind":"bar","x":"category","y":"sales"}', "b"),
      text("Saved runtime/outputs/sum_of_sales_by_category.png: Electronics 2760, Clothing 1880, Grocery 900")]
T3 = [text("Grocery averages 45 units.")]
REPORT = text("# Sales\nTotal sales were 5540; Electronics led with 2760.\n\n"
              "![chart](runtime/outputs/sum_of_sales_by_category.png)")


def run(tmp_path, monkeypatch, responses, **kw):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(responses)
    return client, run_autonomous(GOAL, CsvTools(CSV), client=client, verbose=False,
                                  sleep=lambda s: None, **kw)


def test_goal_to_report(tmp_path, monkeypatch):
    _, state = run(tmp_path, monkeypatch, [PLAN, *T1, *T2, *T3, REPORT])
    assert state["status"] == "done" and set(state["results"]) == {"t1", "t2", "t3"}
    report = (tmp_path / "runtime" / "outputs" / "report.md").read_text()
    assert "5540" in report and "](sum_of_sales_by_category.png)" in report  # link made relative
    assert (tmp_path / "runtime" / "outputs" / "sum_of_sales_by_category.png").exists()
    assert state["totals"]["llm_calls"] == 7


def test_rejected_plan_runs_nothing(tmp_path, monkeypatch):
    client, state = run(tmp_path, monkeypatch, [PLAN], approve_plan=lambda tasks: False)
    assert state["status"] == "plan rejected" and len(client.requests) == 1


def test_budget_stop_then_resume_skips_finished_tasks(tmp_path, monkeypatch):
    _, state = run(tmp_path, monkeypatch, [PLAN, *T1, *T2, *T3, REPORT], budget=Budget(max_llm_calls=3))
    assert state["status"].startswith("stopped") and list(state["results"]) == ["t1"]

    client, state = run(tmp_path, monkeypatch, [*T2, *T3, REPORT])  # second run: no plan, no t1
    assert state["status"] == "done" and len(client.requests) == 4


def test_failed_attempt_is_retried_with_backoff(tmp_path, monkeypatch):
    waits = []
    monkeypatch.chdir(tmp_path)
    client = FakeClient([PLAN, RuntimeError("rate limited"), *T1, *T2, *T3, REPORT])
    state = run_autonomous(GOAL, CsvTools(CSV), client=client, verbose=False, sleep=waits.append)
    assert state["status"] == "done" and waits == [1]
    assert any("attempt 1 failed" in line for line in state["log"])


def test_task_that_keeps_failing_is_marked_and_report_still_written(tmp_path, monkeypatch):
    errors = [RuntimeError("down")] * 3
    # The report may only use numbers from tasks that worked (t1 failed, so no 5540).
    _, state = run(tmp_path, monkeypatch, [PLAN, *errors, *T2, *T3, text("# Sales\nElectronics led with 2760.")])
    assert state["results"]["t1"].startswith("FAILED") and state["status"] == "done"


def test_untraceable_numbers_are_revised_then_flagged(tmp_path, monkeypatch):
    bad = text("# Sales\nTotal sales were 9999.")
    client, state = run(tmp_path, monkeypatch, [PLAN, *T1, *T2, *T3, bad, bad])
    assert "not in the task results" in client.requests[-1][-1]["content"]
    assert "**Unverified:**" in (tmp_path / "runtime" / "outputs" / "report.md").read_text()


def test_plan_rules_are_enforced_by_validation(tmp_path, monkeypatch):
    no_chart = text(json.dumps({"tasks": [t for t in TASKS if t["worker"] != "chart"] + [TASKS[0] | {"id": "t9"}]}))
    dup_ids = text(json.dumps({"tasks": [t | {"id": "t1"} for t in TASKS]}))
    monkeypatch.chdir(tmp_path)
    client = FakeClient([no_chart, dup_ids, PLAN, *T1, *T2, *T3, REPORT])
    state = run_autonomous("Report with a chart", CsvTools(CSV), client=client, verbose=False)
    assert state["status"] == "done"
    feedback = [r[-1]["content"] for r in client.requests[1:3]]
    assert "no task uses the chart worker" in feedback[0] and "unique" in feedback[1]
