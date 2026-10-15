import json
from src.agent import run_agent
from src.memory import ConversationMemory
from src.tools import CsvTools
from tests.fakes import FakeClient, call, text

tools = CsvTools("data/sample_sales.csv")


def test_tool_result_is_fed_back_with_matching_id():
    client = FakeClient([call("aggregate", '{"column":"sales","how":"sum","group_by":"category"}', "abc"),
                         text("Electronics has the highest total.")])
    answer = run_agent("Which category sells most?", tools, client=client, verbose=False)
    assert answer == "Electronics has the highest total."
    second = client.requests[1]
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "abc"
    assert json.loads(second[-1]["content"])["Electronics"] == 2760
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "abc"


def test_unknown_tool_and_bad_args_become_errors_not_crashes():
    client = FakeClient([call("rm_rf", "{}", "a"), call("aggregate", "not json", "b"), text("cannot do that")])
    assert run_agent("q", tools, client=client, verbose=False) == "cannot do that"
    assert "error" in client.requests[1][-1]["content"]
    assert "error" in client.requests[2][-1]["content"]


def test_step_limit_stops_a_looping_model():
    client = FakeClient([call("describe_data", "{}", f"c{i}") for i in range(5)])
    assert run_agent("q", tools, max_steps=3, client=client, verbose=False).startswith("Stopped")
    assert len(client.requests) == 3


def test_memory_carries_over_between_questions():
    mem = ConversationMemory()
    client = FakeClient([text("first answer"), text("second answer")])
    run_agent("first question", tools, memory=mem, client=client, verbose=False)
    run_agent("second question", tools, memory=mem, client=client, verbose=False)
    sent = [m.get("content") for m in client.requests[1]]
    assert "first question" in sent and "first answer" in sent and "second question" in sent


def test_memory_compacts_after_a_long_conversation():
    mem = ConversationMemory(max_tokens=150, keep_turns=1)
    client = FakeClient([text("a" * 300) for _ in range(4)])
    for i in range(4):
        run_agent(f"question {i}", tools, memory=mem, client=client, verbose=False)
    assert mem.summary and len([m for m in mem.messages if m["role"] == "user"]) == 1


def test_step_limit_leaves_history_well_formed():
    mem = ConversationMemory()
    client = FakeClient([call("describe_data", "{}", f"c{i}") for i in range(5)])
    run_agent("q", tools, max_steps=2, memory=mem, client=client, verbose=False)
    called = {tc["id"] for m in mem.messages for tc in m.get("tool_calls", [])}
    returned = {m["tool_call_id"] for m in mem.messages if m["role"] == "tool"}
    assert called == returned and mem.messages[-1]["role"] == "assistant"
