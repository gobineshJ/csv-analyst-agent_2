import pandas as pd
from langchain_core.messages import AIMessage

from eval.run_eval import CASES, REFUSE, grade, parse_trace, run_langgraph, run_plain, to_markdown
from tests.fake_llm import FakeChat, ai_call
from tests.fakes import FakeClient, call, text

df = pd.read_csv("data/sample_sales.csv")
case = {c.id: c for c in CASES}


def test_ground_truth_matches_hand_computed_values():
    assert case["total_sales"].expected(df) == 5540
    assert case["top_category"].expected(df) == ["Electronics", 2760]
    assert case["clothing_north_avg"].expected(df) == 640
    assert case["since_date"].expected(df) == 1840
    assert case["followup_after_compaction"].expected(df) == 1600


def test_grade_numbers_text_and_refusals():
    assert grade(5540, "Total sales are 5,540.")[0]
    assert grade(20.0, "The average is 20.00 units.")[0]
    assert not grade(5540, "Total sales are 5,000.")[0]
    assert grade(["Electronics", 2760], "Electronics leads with $2,760.")[0]
    assert not grade(["Electronics", 2760], "Electronics leads.")[0]  # every part must appear
    assert grade(REFUSE, "I can't predict the future with these tools.")[0]
    assert grade(REFUSE, "I can’t predict that.")[0]  # models often use a curly apostrophe
    assert not grade(REFUSE, "Next month's sales will be 6000.")[0]


def test_parse_trace_counts_calls_and_errors():
    out = ("CALL   describe_data {}\nRESULT {'rows': 8}\n"
           "CALL   aggregate {'column': 'unit'}  (via MCP)\nRESULT {\"error\": \"unknown column\"}\n")
    assert parse_trace(out) == (["describe_data", "aggregate"], 1)


def test_plain_runner_records_and_grades_offline():
    client = FakeClient([call("aggregate", '{"column":"sales","how":"sum"}'), text("Total sales: 5540")])
    [r] = list(run_plain([case["total_sales"]], "data/sample_sales.csv", delay=0, client=client))
    assert r["answer"] == "Total sales: 5540" and r["calls"] == ["aggregate"] and r["tool_errors"] == 0
    r["passed"], r["expected"] = grade(case["total_sales"].expected(df), r["answer"])
    assert r["passed"] and "| PASS |" in to_markdown([r], "data/sample_sales.csv", "fake")


def test_langgraph_runner_offline():
    llm = FakeChat(script=[ai_call("aggregate", {"column": "sales", "how": "sum"}), AIMessage("5540")])
    [r] = list(run_langgraph([case["total_sales"]], "data/sample_sales.csv", delay=0, llm=llm))
    assert r["answer"] == "5540" and r["calls"] == ["aggregate"]


def test_runner_turns_exceptions_into_error_answers():
    client = FakeClient([])  # pop from empty list -> IndexError inside the agent
    [r] = list(run_plain([case["total_sales"]], "data/sample_sales.csv", delay=0, client=client))
    assert r["answer"].startswith("ERROR: IndexError")
