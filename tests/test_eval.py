import json

import pandas as pd
from langchain_core.messages import AIMessage

from eval.run_eval import (CASES, REFUSE, grade, langgraph_case, longmem_case, multi_case, parse_trace,
                           plain_case, plan_case, run_one, structured_case, to_markdown)
from tests.fake_llm import FakeChat, ai_call
from tests.fakes import FakeClient, call, text

df = pd.read_csv("data/sample_sales.csv")
case = {c.id: c for c in CASES}


def test_ground_truth_matches_hand_computed_values():
    assert case["total_sales"].expected(df) == 5540
    assert case["top_category"].expected(df) == ["Electronics", 2760]
    assert case["clothing_north_avg"].expected(df) == 640
    assert case["electronics_share"].expected(df) == 49.8
    assert case["since_date"].expected(df) == 1840
    assert case["followup_after_compaction"].expected(df) == 1600
    assert case["injection"].expected(pd.read_csv(case["injection"].data)) == 5540


def test_grade_numbers_text_refusals_and_forbidden_phrases():
    assert grade(5540, "Total sales are 5,540.")[0]
    assert grade(20.0, "The average is 20.00 units.")[0]
    assert not grade(5540, "Total sales are 5,000.")[0]
    assert grade(5540, "Total sales: 5 540")[0]  # narrow no-break space as thousands separator
    assert grade(REFUSE, "Insufficient data to predict next month's total sales")[0]
    assert grade(["Electronics", 2760], "Electronics leads with $2,760.")[0]
    assert not grade(["Electronics", 2760], "Electronics leads.")[0]  # every part must appear
    assert grade(REFUSE, "I can't predict the future with these tools.")[0]
    assert grade(REFUSE, "I can’t predict that.")[0]  # models often use a curly apostrophe
    assert not grade(REFUSE, "Next month's sales will be 6000.")[0]
    assert not grade(5540, "Total sales are 0. (Actually 5540.)", forbid=["total sales are 0"])[0]


def test_parse_trace_counts_calls_and_errors():
    out = ("CALL   describe_data {}\nRESULT {'rows': 8}\n"
           "CALL   aggregate {'column': 'unit'}  (via MCP)\nRESULT {\"error\": \"unknown column\"}\n")
    assert parse_trace(out) == (["describe_data", "aggregate"], 1)


def test_plain_runner_records_and_grades_offline():
    client = FakeClient([call("aggregate", '{"column":"sales","how":"sum"}'), text("Total sales: 5540")])
    r = run_one("plain", lambda c: plain_case(c, client), case["total_sales"])
    assert r["answer"] == "Total sales: 5540" and r["calls"] == ["aggregate"] and r["tool_errors"] == 0
    r["passed"], r["expected"] = grade(case["total_sales"].expected(df), r["answer"])
    assert r["passed"] and "| PASS |" in to_markdown([r], ["plain"], "fake")


def test_structured_runner_grades_the_json_answer():
    good = {"answerable": True, "answer": "Total is 5540.", "value": 5540, "tools_used": ["aggregate"],
            "confidence": "high"}
    client = FakeClient([call("aggregate", '{"column":"sales","how":"sum"}'), text("5540"), text(json.dumps(good))])
    r = run_one("structured", lambda c: structured_case(c, client), case["total_sales"])
    assert r["answer"] == "Total is 5540." and "value=5540" in r["note"]


def test_langgraph_runner_offline():
    llm = FakeChat(script=[ai_call("aggregate", {"column": "sales", "how": "sum"}), AIMessage("5540")], seen=[])
    r = run_one("langgraph", lambda c: langgraph_case(c, llm), case["total_sales"])
    assert r["answer"] == "5540" and r["calls"] == ["aggregate"]


def test_plan_runner_offline():
    llm = FakeChat(script=[AIMessage('{"steps": ["sum sales"]}'),
                           ai_call("aggregate", {"column": "sales", "how": "sum"}), AIMessage("5540"),
                           AIMessage('{"done": true, "answer": "Total sales are 5540."}')], seen=[])
    r = run_one("plan", lambda c: plan_case(c, llm), case["total_sales"])
    assert r["answer"] == "Total sales are 5540." and r["calls"] == ["aggregate"]


def test_multi_runner_offline():
    client = FakeClient([text('{"tasks": [{"worker": "analyst", "task": "total sales"}]}'), text("5540")])
    r = run_one("multi", lambda c: multi_case(c, client), case["total_sales"])
    assert r["answer"] == "5540" and r["note"] == "specialists: analyst"


def test_longmem_runner_carries_answers_across_sessions():
    c = case["followup_after_compaction"]
    client = FakeClient([text(f"answer {i}") for i in range(len(c.turns))])
    run_one("longmem", lambda c: longmem_case(c, client), c)
    last_system = client.requests[-1][0]["content"]
    assert "Q: What are the total sales in the North region? A: answer 0" in last_system
    assert all(len(req) == 2 for req in client.requests)  # system + question: no conversation history


def test_big_orders_needs_the_data_dictionary():
    assert case["big_orders"].expected(df) == 2160


def test_runner_turns_exceptions_into_error_answers():
    client = FakeClient([])  # pop from empty list -> IndexError inside the agent
    r = run_one("plain", lambda c: plain_case(c, client), case["total_sales"])
    assert r["answer"].startswith("ERROR: IndexError")
