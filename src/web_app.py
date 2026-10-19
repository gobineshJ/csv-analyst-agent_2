"""Stage 10: the agent as a web app (FastAPI + one HTML page).

    python main_web.py            then open http://127.0.0.1:8000

What changes when an agent leaves the terminal:

- No blocking for a human. The CLI waits on input() for an approval; a web server can't hold a
  request open while someone thinks. So the LangGraph agent pauses at interrupt(), its state is
  saved in SQLite (web.db), the HTTP response ends with an "approval" event, and a LATER request
  (POST /decide) resumes it. This is why stage 5 used interrupt() and stage 8 added SqliteSaver.
- Streaming. Agent runs take seconds, so events (tool calls, results, the answer) are sent as
  they happen using Server-Sent Events, from the same stream_events() the CLI prints.
- Sessions. Each conversation is a LangGraph thread. Reload the page or restart the server and
  the history and any pending approval are still there.
- Untrusted input from the network. The browser picks a CSV by name from a fixed list (data/*.csv)
  and never sends a path, so it can't read other files on the machine.

Local learning app only: there is no login, so it listens on 127.0.0.1 and must not be exposed.
"""
import json
import sqlite3
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agent import MODEL
from src.langgraph_agent import build_graph, make_llm, sqlite_checkpointer, stream_events
from src.tools import CsvTools
from src.tracing import Tracer, TracingCallback

PAGE = Path(__file__).parent / "static" / "index.html"


class NewSession(BaseModel):
    csv: str


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Decision(BaseModel):
    action: str = Field(pattern="^(approve|edit|reject)$")
    code: str | None = None
    reason: str | None = None


def create_app(data_dir="data", db_path="web.db", llm=None):
    """llm: a LangChain chat model (tests pass a fake); default is ChatGroq."""
    app = FastAPI(title="CSV Analyst Agent")
    datasets = {p.name: p for p in sorted(Path(data_dir).glob("*.csv"))}
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, csv TEXT, title TEXT, created REAL)")
    checkpointer = sqlite_checkpointer(db_path)
    model = llm or make_llm()
    graphs = {}  # one compiled graph per CSV; all share the checkpointer, threads keep them apart

    def graph_for(csv_name):
        if csv_name not in graphs:
            graphs[csv_name] = build_graph(CsvTools(str(datasets[csv_name])), model, approve_code=True,
                                           checkpointer=checkpointer)
        return graphs[csv_name]

    def session(session_id):
        row = db.execute("SELECT id, csv, title, created FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            raise HTTPException(404, "unknown session")
        return dict(zip(("id", "csv", "title", "created"), row))

    def config(session_id):
        return {"configurable": {"thread_id": session_id}}

    def run(session_id, inputs):
        """Stream one graph run as Server-Sent Events, then token totals, then done."""
        graph = graph_for(session(session_id)["csv"])
        tracer = Tracer(write=False)

        def events():
            try:
                yield from stream_events(graph, inputs, session_id, callbacks=[TracingCallback(tracer, MODEL)])
            except Exception as e:  # report it to the page instead of dropping the connection
                yield {"type": "error", "message": f"{type(e).__name__}: {str(e)[:300]}"}
            yield {"type": "usage", **tracer.totals()}
            yield {"type": "done"}

        body = (f"data: {json.dumps(e, default=str)}\n\n" for e in events())
        return StreamingResponse(body, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/")
    def page():
        return FileResponse(PAGE)

    @app.get("/api/datasets")
    def list_datasets():
        return list(datasets)

    @app.get("/api/sessions")
    def list_sessions():
        rows = db.execute("SELECT id, csv, title, created FROM sessions ORDER BY created DESC").fetchall()
        return [dict(zip(("id", "csv", "title", "created"), r)) for r in rows]

    @app.post("/api/sessions")
    def new_session(body: NewSession):
        if body.csv not in datasets:
            raise HTTPException(400, f"unknown dataset; choose one of {list(datasets)}")
        sid = uuid.uuid4().hex
        db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", (sid, body.csv, "New conversation", time.time()))
        db.commit()
        return session(sid)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        info = session(session_id)
        state = graph_for(info["csv"]).get_state(config(session_id))
        messages = []
        for m in state.values.get("messages", []):
            if m.type == "human":
                messages.append({"role": "user", "text": m.content})
            elif m.type == "ai":
                messages += [{"role": "call", "name": tc["name"], "args": tc["args"]} for tc in m.tool_calls]
                if m.content and not m.tool_calls:
                    messages.append({"role": "assistant", "text": m.content})
            elif m.type == "tool":
                messages.append({"role": "result", "name": m.name, "content": str(m.content)})
        return {**info, "messages": messages, "pending": [i.value for i in state.interrupts]}

    @app.post("/api/sessions/{session_id}/ask")
    def ask(session_id: str, body: Ask):
        info = session(session_id)
        if graph_for(info["csv"]).get_state(config(session_id)).interrupts:
            raise HTTPException(409, "this conversation is waiting for an approval decision")
        if info["title"] == "New conversation":
            db.execute("UPDATE sessions SET title = ? WHERE id = ?", (body.question[:60], session_id))
            db.commit()
        return run(session_id, {"messages": [("user", body.question)]})

    @app.post("/api/sessions/{session_id}/decide")
    def decide(session_id: str, body: Decision):
        info = session(session_id)
        pending = graph_for(info["csv"]).get_state(config(session_id)).interrupts
        if not pending:
            raise HTTPException(409, "nothing is waiting for approval")
        decision = {"action": body.action, "reason": body.reason or "not approved"}
        if body.action == "edit":
            decision["args"] = {**pending[0].value["args"], "code": body.code or ""}
        return run(session_id, Command(resume=decision))

    return app
