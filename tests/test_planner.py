import json

from langchain_core.messages import AIMessage

from src.core.tools import CsvTools
from src.patterns.planner import build_planner, run_planner
from tests.fakes import FakeChat, ai_call

csv = CsvTools("data/sample_sales.csv")


def j(obj):
    return AIMessage(content=json.dumps(obj))


def test_plan_execute_replan_answer():
    llm = FakeChat(script=[
        j({"steps": ["Sum sales by region"]}),                                          # planner
        ai_call("aggregate", {"column": "sales", "how": "sum", "group_by": "region"}),  # executor
        AIMessage(content="North 3940, South 1600"),
        j({"done": True, "answer": "North has more sales: 3940 vs 1600."}),             # replanner
    ], seen=[])
    answer = run_planner("Which region sells more?", build_planner(csv, llm, verbose=False))
    assert answer == "North has more sales: 3940 vs 1600."
    assert "3940" in str(llm.seen[2])  # the executor saw the tool result


def test_replanner_can_add_steps_and_invalid_json_is_retried():
    llm = FakeChat(script=[
        AIMessage(content='```json\n{"steps": []}\n```'),  # fenced, and too few steps -> retried
        j({"steps": ["Describe the data"]}),
        AIMessage(content="There are 8 rows."),
        j({"done": False, "steps": ["Count rows with units > 10"]}),
        AIMessage(content="5 rows."),
        j({"done": True, "answer": "5"}),
    ], seen=[])
    assert run_planner("q", build_planner(csv, llm, verbose=False)) == "5"


def test_plan_step_limit():
    llm = FakeChat(script=[j({"steps": ["a"]})] + [AIMessage(content="r"), j({"done": False, "steps": ["again"]})] * 3,
                   seen=[])
    assert run_planner("q", build_planner(csv, llm, max_steps=2, verbose=False)).startswith("Stopped")
