import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.web_app import create_app
from tests.fake_llm import FakeChat, ai_call

CODE = "result = len(df)"


def events(resp):
    assert resp.status_code == 200, resp.text
    return [json.loads(line[6:]) for line in resp.text.split("\n\n") if line.startswith("data: ")]


@pytest.fixture
def make(tmp_path):
    def build(script):
        llm = FakeChat(script=script, seen=[])
        return llm, TestClient(create_app(db_path=str(tmp_path / "web.db"), llm=llm))
    return build


def new_session(client, csv="sample_sales.csv"):
    return client.post("/api/sessions", json={"csv": csv}).json()["id"]


def test_page_and_datasets(make):
    _, client = make([])
    assert "CSV Analyst Agent" in client.get("/").text
    assert "sample_sales.csv" in client.get("/api/datasets").json()


def test_ask_streams_tool_calls_answer_and_usage(make):
    _, client = make([ai_call("aggregate", {"column": "sales", "how": "sum"}), AIMessage(content="Total is 5540.")])
    sid = new_session(client)
    got = events(client.post(f"/api/sessions/{sid}/ask", json={"question": "total?"}))
    assert [e["type"] for e in got] == ["call", "result", "answer", "usage", "done"]
    assert "5540" in got[1]["content"] and got[2]["text"] == "Total is 5540."
    assert got[3]["llm_calls"] == 2  # traced through LangChain callbacks
    assert client.get("/api/sessions").json()[0]["title"] == "total?"


def test_code_pauses_for_approval_across_requests(make):
    llm, client = make([ai_call("run_pandas", {"code": CODE}), AIMessage(content="There are 8 rows.")])
    sid = new_session(client)
    first = events(client.post(f"/api/sessions/{sid}/ask", json={"question": "rows?"}))
    assert first[-3]["type"] == "approval" and first[-3]["request"]["args"]["code"] == CODE

    # The HTTP request ended; the pause lives in SQLite. A reload sees it, and new questions are refused.
    assert client.get(f"/api/sessions/{sid}").json()["pending"][0]["args"]["code"] == CODE
    assert client.post(f"/api/sessions/{sid}/ask", json={"question": "other"}).status_code == 409

    second = events(client.post(f"/api/sessions/{sid}/decide", json={"action": "approve"}))
    assert [e["type"] for e in second][:2] == ["result", "answer"]
    assert json.loads(second[0]["content"]) == {"result": 8}
    assert client.get(f"/api/sessions/{sid}").json()["pending"] == []


def test_edit_and_reject_decisions(make):
    llm, client = make([ai_call("run_pandas", {"code": CODE}), AIMessage(content="ok"),
                        ai_call("run_pandas", {"code": CODE}, "c2"), AIMessage(content="fine")])
    sid = new_session(client)
    client.post(f"/api/sessions/{sid}/ask", json={"question": "q1"})
    edited = events(client.post(f"/api/sessions/{sid}/decide", json={"action": "edit", "code": "result = 42"}))
    assert json.loads(edited[0]["content"]) == {"result": 42}

    client.post(f"/api/sessions/{sid}/ask", json={"question": "q2"})
    rejected = events(client.post(f"/api/sessions/{sid}/decide", json={"action": "reject", "reason": "no"}))
    assert "rejected" in rejected[0]["content"]


def test_history_survives_a_server_restart(tmp_path):
    db = str(tmp_path / "web.db")
    client = TestClient(create_app(db_path=db, llm=FakeChat(script=[AIMessage(content="hi")], seen=[])))
    sid = new_session(client)
    client.post(f"/api/sessions/{sid}/ask", json={"question": "hello"})

    restarted = TestClient(create_app(db_path=db, llm=FakeChat(script=[], seen=[])))
    messages = restarted.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_only_listed_datasets_and_known_sessions(make):
    _, client = make([])
    assert client.post("/api/sessions", json={"csv": "../.env"}).status_code == 400
    assert client.post("/api/sessions", json={"csv": "C:/Windows/win.ini"}).status_code == 400
    assert client.get("/api/sessions/nope").status_code == 404
    sid = new_session(client)
    assert client.post(f"/api/sessions/{sid}/decide", json={"action": "approve"}).status_code == 409
    assert client.post(f"/api/sessions/{sid}/decide", json={"action": "delete"}).status_code == 422


def test_model_errors_are_reported_not_dropped(make):
    _, client = make([])  # empty script: the fake model raises
    sid = new_session(client)
    got = events(client.post(f"/api/sessions/{sid}/ask", json={"question": "q"}))
    assert got[0]["type"] == "error" and got[-1]["type"] == "done"
