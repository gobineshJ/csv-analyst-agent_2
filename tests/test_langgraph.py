from langchain_core.messages import AIMessage

from src.core.tools import CsvTools
from src.frameworks.langgraph_agent import build_graph, run_agent_langgraph
from tests.fakes import FakeChat, ai_call

csv = CsvTools("data/sample_sales.csv")


def make(script, **kw):
    llm = FakeChat(script=script, seen=[])
    return llm, build_graph(csv, llm, **kw)


def assert_no_dangling_tool_calls(messages):
    called = {tc["id"] for m in messages if m.type == "ai" for tc in m.tool_calls}
    returned = {m.tool_call_id for m in messages if m.type == "tool"}
    assert called <= returned, called - returned


def test_tool_runs_and_result_is_fed_back():
    llm, g = make([ai_call("aggregate", {"column": "sales", "how": "sum", "group_by": "category"}),
                   AIMessage(content="Electronics")])
    assert run_agent_langgraph("q", g, verbose=False) == "Electronics"
    tool_msgs = [m for m in llm.seen[1] if m.type == "tool"]
    assert len(tool_msgs) == 1 and "2760" in tool_msgs[0].content


def test_memory_is_per_thread():
    llm, g = make([AIMessage(content="a1"), AIMessage(content="a2"), AIMessage(content="a3")])
    run_agent_langgraph("first question", g, thread_id="t1", verbose=False)
    run_agent_langgraph("second question", g, thread_id="t1", verbose=False)
    run_agent_langgraph("other thread", g, thread_id="t2", verbose=False)
    assert "first question" in [m.content for m in llm.seen[1]]
    assert "first question" not in [m.content for m in llm.seen[2]]


def test_trimming_hides_old_turns_from_model_but_keeps_state():
    llm, g = make([AIMessage(content="x" * 400) for _ in range(5)], max_context_tokens=300)
    for i in range(5):
        run_agent_langgraph(f"question {i}", g, verbose=False)
    sent = llm.seen[-1]
    assert sent[0].type == "system" and sent[1].type == "human"  # starts on a human message
    assert "question 0" not in [m.content for m in sent]
    full = g.get_state({"configurable": {"thread_id": "default"}}).values["messages"]
    assert "question 0" in [m.content for m in full]


def test_latest_turn_over_budget_is_still_sent():
    llm, g = make([AIMessage(content="ok")], max_context_tokens=1)
    assert run_agent_langgraph("a fairly long question here", g, verbose=False) == "ok"
    assert any(m.content == "a fairly long question here" for m in llm.seen[0])


def test_step_limit_then_followup_has_no_dangling_tool_call():
    llm, g = make([ai_call("describe_data", {}, f"c{i}") for i in range(10)] + [AIMessage(content="fine")])
    assert run_agent_langgraph("loop", g, max_steps=2, verbose=False).startswith("Stopped")
    llm.script[:] = [AIMessage(content="recovered")]
    assert run_agent_langgraph("next question", g, verbose=False) == "recovered"
    assert_no_dangling_tool_calls(llm.seen[-1])


def test_malformed_tool_call_rejected_by_provider_is_retried():
    class Flaky(FakeChat):
        failed: bool = False

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if not self.failed:
                self.failed = True
                raise RuntimeError("Error code: 400 - tool_use_failed")
            return super()._generate(messages, stop, run_manager, **kw)

    llm = Flaky(script=[AIMessage(content="ok")], seen=[])
    assert run_agent_langgraph("q", build_graph(csv, llm), verbose=False) == "ok"
    assert "not valid JSON" in llm.seen[0][-1].content


def test_unknown_tool_does_not_crash():
    llm, g = make([ai_call("rm_rf", {}), AIMessage(content="cannot")])
    assert run_agent_langgraph("q", g, verbose=False) == "cannot"
    assert "error" in [m for m in llm.seen[1] if m.type == "tool"][0].content.lower()
