import json
from types import SimpleNamespace as NS

import pytest
from langchain_core.messages import AIMessage

from src.core.agent import run_agent
from src.core.tools import CsvTools
from src.frameworks.langgraph_agent import build_graph, run_agent_langgraph
from src.patterns.tracing import Budget, BudgetExceeded, TracedClient, TracedTools, Tracer, TracingCallback, summarize
from tests.fakes import FakeChat, FakeClient, ai_call, call, text


def with_usage(resp, prompt, completion):
    resp.usage = NS(prompt_tokens=prompt, completion_tokens=completion)
    return resp


def test_records_llm_and_tool_calls_with_tokens_and_cost(tmp_path):
    tracer = Tracer(out_dir=tmp_path)
    client = TracedClient(FakeClient([with_usage(call("aggregate", '{"column":"sales","how":"sum"}'), 1000, 50),
                                      with_usage(text("5540"), 1200, 10)]), tracer)
    tools = TracedTools(CsvTools("data/sample_sales.csv"), tracer)
    assert run_agent("total?", tools, client=client, verbose=False) == "5540"

    t = tracer.totals()
    assert (t["llm_calls"], t["tool_calls"], t["input_tokens"], t["output_tokens"]) == (2, 1, 2200, 60)
    assert t["cost_usd"] > 0
    events = [json.loads(line) for line in tracer.path.read_text().splitlines()]
    assert [e["kind"] for e in events] == ["llm", "tool", "llm"]
    assert events[0]["tool_calls"] == ["aggregate"] and events[1]["name"] == "aggregate"


def test_tool_errors_and_attributes_pass_through():
    tracer = Tracer(write=False)
    tools = TracedTools(CsvTools("data/sample_sales.csv"), tracer)
    assert tools.path == "data/sample_sales.csv"
    tools.aggregate("nope", "sum")
    assert tracer.totals()["tool_errors"] == 1


def test_failed_llm_call_is_recorded_then_raised():
    tracer = Tracer(write=False)
    client = TracedClient(FakeClient([RuntimeError("boom")]), tracer)
    with pytest.raises(RuntimeError):
        client.chat.completions.create(model="m", messages=[])
    assert tracer.events[0]["error"] == "boom"


def test_budget_stops_further_llm_calls():
    tracer = Tracer(write=False, budget=Budget(max_llm_calls=1))
    client = TracedClient(FakeClient([call("describe_data", "{}"), text("never reached")]), tracer)
    with pytest.raises(BudgetExceeded):
        run_agent("q", CsvTools("data/sample_sales.csv"), client=client, verbose=False)


def test_langgraph_callback_records_model_calls():
    tracer = Tracer(write=False)
    llm = FakeChat(script=[ai_call("describe_data", {}), AIMessage(content="done")], seen=[],
                   callbacks=[TracingCallback(tracer, "fake")])
    run_agent_langgraph("q", build_graph(TracedTools(CsvTools("data/sample_sales.csv"), tracer), llm), verbose=False)
    assert [e["kind"] for e in tracer.events] == ["llm", "tool", "llm"]


def test_summarize_prints_timeline(tmp_path, capsys):
    tracer = Tracer(out_dir=tmp_path)
    TracedClient(FakeClient([with_usage(text("hi"), 10, 2)]), tracer).chat.completions.create(model="m", messages=[])
    summarize(tracer.path)
    assert '"llm_calls": 1' in capsys.readouterr().out
