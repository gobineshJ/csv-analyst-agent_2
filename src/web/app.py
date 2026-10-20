"""Stage 10: the agent as a web app (FastAPI + a static page in src/web/static/).

    python -m apps.web            then open http://127.0.0.1:8000

What changes when an agent leaves the terminal:

- No blocking for a human. The CLI waits on input() for an approval; a web server can't hold a
  request open while someone thinks. So the LangGraph agent pauses at interrupt(), its state is
  saved in SQLite (runtime/web.db), the HTTP response ends with an "approval" event, and a LATER
  request (POST /decide) resumes it. This is why stage 5 used interrupt() and stage 8 added SqliteSaver.
- Streaming. Agent runs take seconds, so events (tool calls, results, the answer) are sent as
  they happen using Server-Sent Events, from the same stream_events() the CLI prints.
- Sessions. Each conversation is a LangGraph thread. Reload the page or restart the server and
  the history and any pending approval are still there.
- Untrusted input from the network. Every value the browser sends is checked: datasets are picked
  by name from a list, uploads are size-limited and must parse as CSV, and files are only served
  from runtime/outputs. Nothing the browser sends is ever used as a path.

Local learning app only: there is no login, so it listens on 127.0.0.1 and must not be exposed.
"""
import io
import json
import queue
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.core.agent import MODEL
from src.core.tools import CsvTools
from src.frameworks.langgraph_agent import build_graph, make_llm, sqlite_checkpointer, stream_events
from src.paths import OUTPUTS, RUNTIME, runtime_file
from src.patterns.autonomous import run_autonomous
from src.patterns.hitl import auto_approve
from src.patterns.tracing import Budget, Tracer, TracingCallback

STATIC = Path(__file__).parent / "static"
UPLOADS = RUNTIME / "uploads"
MAX_UPLOAD_CHARS = 5_000_000
MAX_COLUMNS = 100


class NewSession(BaseModel):
    csv: str


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Decision(BaseModel):
    action: str = Field(pattern="^(approve|edit|reject)$")
    code: str | None = None
    reason: str | None = None


class Upload(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=MAX_UPLOAD_CHARS)


class ReportRequest(BaseModel):
    csv: str
    goal: str = Field(min_length=5, max_length=500)


def sse(events):
    """Turn a stream of event dicts into a Server-Sent Events response."""
    body = (f"data: {json.dumps(e, default=str)}\n\n" for e in events)
    return StreamingResponse(body, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def safe_csv_name(name):
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")[:50]
    return (stem or "data") + ".csv"


def create_app(data_dir="data", db_path=None, llm=None, client=None):
    """llm: LangChain chat model for conversations; client: Groq client for reports.
    Both default to the real Groq model; tests pass fakes."""
    app = FastAPI(title="CSV Analyst Agent")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    db_path = db_path or runtime_file("web.db")
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, csv TEXT, title TEXT, created REAL)")
    checkpointer = sqlite_checkpointer(db_path)
    model = llm or make_llm()
    graphs = {}  # one compiled graph per CSV; all share the checkpointer, threads keep them apart

    def datasets():
        """Name -> path for the bundled CSVs and the uploaded ones. Names are the only thing the page sends."""
        found = {p.name: p for p in sorted(UPLOADS.glob("*.csv"))}
        found.update({p.name: p for p in sorted(Path(data_dir).glob("*.csv"))})
        return found

    def dataset_path(name):
        if (path := datasets().get(name)) is None:
            raise HTTPException(400, f"unknown dataset '{name}'")
        return path

    def graph_for(csv_name):
        if csv_name not in graphs:
            graphs[csv_name] = build_graph(CsvTools(str(dataset_path(csv_name))), model, approve_code=True,
                                           checkpointer=checkpointer, charts=True)
        return graphs[csv_name]

    def session(session_id):
        row = db.execute("SELECT id, csv, title, created FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            raise HTTPException(404, "unknown session")
        return dict(zip(("id", "csv", "title", "created"), row))

    def config(session_id):
        return {"configurable": {"thread_id": session_id}}

    def run(session_id, inputs):
        """Stream one graph run, then token totals, then done."""
        graph = graph_for(session(session_id)["csv"])
        tracer = Tracer(write=False)

        def events():
            try:
                yield from stream_events(graph, inputs, session_id, callbacks=[TracingCallback(tracer, MODEL)])
            except Exception as e:  # report it to the page instead of dropping the connection
                yield {"type": "error", "message": f"{type(e).__name__}: {str(e)[:300]}"}
            yield {"type": "usage", **tracer.totals()}
            yield {"type": "done"}
        return sse(events())

    @app.get("/")
    def page():
        return FileResponse(STATIC / "index.html")

    # --- datasets ---------------------------------------------------------------------

    @app.get("/api/datasets")
    def list_datasets():
        return [{"name": name, "uploaded": path.parent.resolve() == UPLOADS.resolve()}
                for name, path in datasets().items()]

    @app.get("/api/datasets/{name}")
    def preview_dataset(name: str):
        df = pd.read_csv(dataset_path(name))
        return {"name": name, "rows": len(df), "columns": {c: str(t) for c, t in df.dtypes.items()},
                "sample": json.loads(df.head(10).to_json(orient="records"))}

    @app.post("/api/datasets")
    def upload_dataset(body: Upload):
        try:
            df = pd.read_csv(io.StringIO(body.content))
        except Exception as e:
            raise HTTPException(400, f"not a readable CSV: {str(e)[:200]}")
        if df.empty or len(df.columns) > MAX_COLUMNS:
            raise HTTPException(400, f"the CSV must have at least one row and at most {MAX_COLUMNS} columns")
        name, existing = safe_csv_name(body.name), datasets()
        stem, n = name[:-4], 2
        while name in existing:
            name, n = f"{stem}_{n}.csv", n + 1
        UPLOADS.mkdir(parents=True, exist_ok=True)
        (UPLOADS / name).write_text(body.content, encoding="utf-8")
        return {"name": name, "rows": len(df), "columns": len(df.columns)}

    # --- files the agent made (charts) --------------------------------------------------

    @app.get("/api/files/{name}")
    def output_file(name: str):
        path = (OUTPUTS / name).resolve()
        if not re.fullmatch(r"[\w.-]+\.png", name) or path.parent != OUTPUTS.resolve() or not path.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(path)

    # --- conversations ----------------------------------------------------------------

    @app.get("/api/sessions")
    def list_sessions():
        rows = db.execute("SELECT id, csv, title, created FROM sessions ORDER BY created DESC").fetchall()
        return [dict(zip(("id", "csv", "title", "created"), r)) for r in rows]

    @app.post("/api/sessions")
    def new_session(body: NewSession):
        dataset_path(body.csv)
        sid = uuid.uuid4().hex
        db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", (sid, body.csv, "New conversation", time.time()))
        db.commit()
        return session(sid)

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        session(session_id)
        checkpointer.delete_thread(session_id)
        db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        db.commit()
        return {"deleted": session_id}

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

    # --- autonomous reports (stage 9) ---------------------------------------------------

    @app.post("/api/reports")
    def report(body: ReportRequest):
        """Run the autonomous agent in a background thread and stream its progress.
        In the terminal you approve the plan; here it is shown and auto-approved, and a budget caps the run."""
        path = dataset_path(body.csv)
        events = queue.Queue()

        def show_plan(tasks):
            events.put({"type": "plan", "tasks": [t.model_dump() for t in tasks]})
            return True

        def work():
            try:
                state = run_autonomous(
                    body.goal, CsvTools(str(path)), client=client, out_dir=OUTPUTS / "reports" / uuid.uuid4().hex,
                    budget=Budget(max_llm_calls=40, max_seconds=300), approve_plan=show_plan, approve=auto_approve,
                    verbose=False, on_log=lambda m: events.put({"type": "log", "message": m}))
                markdown = Path(state["report_path"]).read_text(encoding="utf-8") if state.get("report_path") else ""
                events.put({"type": "report", "status": state["status"], "markdown": markdown,
                            "totals": state.get("totals", {})})
            except Exception as e:
                events.put({"type": "error", "message": f"{type(e).__name__}: {str(e)[:300]}"})
            events.put(None)

        threading.Thread(target=work, daemon=True).start()

        def stream():
            while (e := events.get()) is not None:
                yield e
            yield {"type": "done"}
        return sse(stream())

    return app
