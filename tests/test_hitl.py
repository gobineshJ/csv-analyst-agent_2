import json

from langchain_core.messages import AIMessage

from src.agent import run_agent
from src.hitl import apply_decision, auto_approve
from src.langgraph_agent import build_graph, resume_langgraph, run_agent_langgraph
from src.tools import CsvTools
from tests.fake_llm import FakeChat, ai_call
from tests.fakes import FakeClient, call, text

csv = CsvTools("data/sample_sales.csv")
CODE = "result = len(df)"


def recorder(decision):
    seen = []

    def approve(request):
        seen.append(request)
        return decision
    return approve, seen


def test_apply_decision():
    assert apply_decision({"action": "approve"}, {"code": "a"}) == ({"code": "a"}, None)
    assert apply_decision({"action": "edit", "args": {"code": "b"}}, {"code": "a"}) == ({"code": "b"}, None)
    args, msg = apply_decision({"action": "reject", "reason": "too risky"}, {"code": "a"})
    assert args is None and "too risky" in msg


# --- plain loop: the approver is a function call that blocks ---------------------------

def test_plain_loop_asks_only_for_risky_tools():
    approve, seen = recorder({"action": "approve"})
    client = FakeClient([call("describe_data", "{}", "a"), call("run_pandas", json.dumps({"code": CODE}), "b"), text("8")])
    run_agent("q", csv, client=client, verbose=False, approve=approve)
    assert [r["tool"] for r in seen] == ["run_pandas"]
    assert json.loads(client.requests[2][-1]["content"]) == {"result": 8}


def test_plain_loop_rejection_goes_back_to_model():
    approve, _ = recorder({"action": "reject", "reason": "no code please"})
    client = FakeClient([call("run_pandas", json.dumps({"code": CODE})), text("ok")])
    run_agent("q", csv, client=client, verbose=False, approve=approve)
    assert "no code please" in client.requests[1][-1]["content"]


def test_plain_loop_edit_runs_the_edited_code():
    approve, _ = recorder({"action": "edit", "args": {"code": "result = 42"}})
    client = FakeClient([call("run_pandas", json.dumps({"code": CODE})), text("ok")])
    run_agent("q", csv, client=client, verbose=False, approve=approve)
    assert json.loads(client.requests[1][-1]["content"]) == {"result": 42}


# --- LangGraph: interrupt() pauses the graph; state waits in the checkpointer -----------

def make(script):
    llm = FakeChat(script=script, seen=[])
    return llm, build_graph(csv, llm, approve_code=True)


def test_langgraph_pauses_and_resumes_later():
    llm, g = make([ai_call("run_pandas", {"code": CODE}), AIMessage(content="There are 8 rows.")])
    first = run_agent_langgraph("how many rows?", g, thread_id="t", verbose=False)  # no ask: just pause
    assert first.startswith("Waiting for approval") and CODE in first
    state = g.get_state({"configurable": {"thread_id": "t"}})
    assert state.interrupts and state.interrupts[0].value["args"]["code"] == CODE
    assert len(llm.seen) == 1  # the model has not been called again while paused

    # ...later, possibly from another process sharing the checkpointer:
    assert resume_langgraph(g, {"action": "approve"}, thread_id="t", verbose=False) == "There are 8 rows."
    tool_msg = [m for m in llm.seen[1] if m.type == "tool"][0]
    assert json.loads(tool_msg.content) == {"result": 8}


def test_langgraph_ask_callback_reject():
    llm, g = make([ai_call("run_pandas", {"code": CODE}), AIMessage(content="I won't run code.")])
    approve, seen = recorder({"action": "reject", "reason": "not today"})
    assert run_agent_langgraph("q", g, verbose=False, ask=approve) == "I won't run code."
    assert seen[0]["tool"] == "run_pandas"
    assert "not today" in [m for m in llm.seen[1] if m.type == "tool"][0].content


def test_langgraph_without_approval_runs_directly():
    llm = FakeChat(script=[ai_call("run_pandas", {"code": CODE}), AIMessage(content="8")], seen=[])
    assert run_agent_langgraph("q", build_graph(csv, llm), verbose=False, ask=auto_approve) == "8"
