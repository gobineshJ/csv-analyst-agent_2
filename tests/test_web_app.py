import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.web.app import create_app, safe_csv_name
from tests.fakes import FakeChat, FakeClient, ai_call, text

CODE = "result = len(df)"
DATA = Path(__file__).resolve().parent.parent / "data"


def events(resp):
    assert resp.status_code == 200, resp.text
    return [json.loads(line[6:]) for line in resp.text.split("\n\n") if line.startswith("data: ")]


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # uploads and charts go to tmp_path/runtime, not the real one

    def build(script, client=None):
        llm = FakeChat(script=script, seen=[])
        return llm, TestClient(create_app(data_dir=DATA, db_path=str(tmp_path / "web.db"), llm=llm, client=client))
    return build


def new_session(client, csv="sample_sales.csv"):
    return client.post("/api/sessions", json={"csv": csv}).json()["id"]


def test_page_assets_and_datasets(make):
    _, client = make([])
    assert "CSV Analyst" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    names = [d["name"] for d in client.get("/api/datasets").json()]
    assert "sample_sales.csv" in names


def test_preview(make):
    _, client = make([])
    p = client.get("/api/datasets/sample_sales.csv").json()
    assert p["rows"] == 8 and "sales" in p["columns"] and len(p["sample"]) == 8
    assert client.get("/api/datasets/nope.csv").status_code == 400


def test_ask_streams_tool_calls_answer_and_usage(make):
    _, client = make([ai_call("aggregate", {"column": "sales", "how": "sum"}), AIMessage(content="Total is 5540.")])
    sid = new_session(client)
    got = events(client.post(f"/api/sessions/{sid}/ask", json={"question": "total?"}))
    assert [e["type"] for e in got] == ["call", "result", "answer", "usage", "done"]
    assert "5540" in got[1]["content"] and got[2]["text"] == "Total is 5540."
    assert got[3]["llm_calls"] == 2  # traced through LangChain callbacks
    assert client.get("/api/sessions").json()[0]["title"] == "total?"


def test_chart_is_made_and_served(make):
    _, client = make([ai_call("make_chart", {"kind": "bar", "x": "category", "y": "sales"}), AIMessage(content="Done.")])
    sid = new_session(client)
    got = events(client.post(f"/api/sessions/{sid}/ask", json={"question": "chart please"}))
    path = json.loads(got[1]["content"])["path"]
    resp = client.get(f"/api/files/{Path(path).name}")
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"


def test_files_endpoint_serves_only_pngs_from_outputs(make, tmp_path):
    _, client = make([])
    (tmp_path / "secret.png").write_bytes(b"x")
    (tmp_path / "runtime" / "outputs").mkdir(parents=True)
    (tmp_path / "runtime" / "outputs" / "notes.txt").write_text("x")
    for name in ["../../secret.png", "..%2F..%2Fsecret.png", "notes.txt", "missing.png", "..\\secret.png"]:
        assert client.get(f"/api/files/{name}").status_code == 404, name


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


def test_delete_session_removes_it_and_its_history(make):
    _, client = make([AIMessage(content="hi")])
    sid = new_session(client)
    client.post(f"/api/sessions/{sid}/ask", json={"question": "hello"})
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.get("/api/sessions").json() == []


def test_history_survives_a_server_restart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / "web.db")
    client = TestClient(create_app(data_dir=DATA, db_path=db, llm=FakeChat(script=[AIMessage(content="hi")], seen=[])))
    sid = new_session(client)
    client.post(f"/api/sessions/{sid}/ask", json={"question": "hello"})

    restarted = TestClient(create_app(data_dir=DATA, db_path=db, llm=FakeChat(script=[], seen=[])))
    messages = restarted.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_upload_validates_and_sanitizes(make):
    _, client = make([])
    ok = client.post("/api/datasets", json={"name": "../../My Sales!.csv", "content": "a,b\n1,2\n3,4\n"}).json()
    assert ok == {"name": "My_Sales.csv", "rows": 2, "columns": 2}
    again = client.post("/api/datasets", json={"name": "My Sales.csv", "content": "a\n1\n"}).json()
    assert again["name"] == "My_Sales_2.csv"  # never overwrites
    shadow = client.post("/api/datasets", json={"name": "sample_sales.csv", "content": "a\n1\n"}).json()
    assert shadow["name"] == "sample_sales_2.csv"  # can't replace a bundled dataset
    assert client.post("/api/datasets", json={"name": "x.csv", "content": "a,b\n"}).status_code == 400  # no rows
    wide = ",".join(f"c{i}" for i in range(101)) + "\n" + ",".join("1" * 101) + "\n"
    assert client.post("/api/datasets", json={"name": "x.csv", "content": wide}).status_code == 400
    names = [d["name"] for d in client.get("/api/datasets").json()]
    assert "My_Sales.csv" in names
    assert new_session(client, "My_Sales.csv")


def test_safe_csv_name():
    assert safe_csv_name("C:/Users/x/data.csv") == "data.csv"
    assert safe_csv_name("....csv") == "data.csv"
    assert safe_csv_name("a b/c.csv") == "c.csv"


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


def test_report_streams_plan_progress_and_markdown(make):
    plan = text(json.dumps({"tasks": [{"id": "t1", "worker": "analyst", "task": "Total sales"},
                                      {"id": "t2", "worker": "analyst", "task": "Sales by region"},
                                      {"id": "t3", "worker": "analyst", "task": "Units by category"}]}))
    fake = FakeClient([plan, text("Total sales are 5540."), text("North 3940, South 1600."),
                       text("Grocery 100 units."), text("# Report\nTotal sales were 5540.")])
    _, client = make([], client=fake)
    got = events(client.post("/api/reports", json={"csv": "sample_sales.csv", "goal": "A short report"}))
    types = [e["type"] for e in got]
    assert types[0] == "plan" and types[-2:] == ["report", "done"] and "log" in types
    assert got[-2]["status"] == "done" and "5540" in got[-2]["markdown"]
