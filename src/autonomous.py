"""Autonomy: give the agent a goal, not a question, and let it work through many steps alone.

    goal -> plan tasks -> [approve plan?] -> run each task (analyst / chart specialist)
         -> write report -> check every number against the task results -> outputs/report.md

An agent that runs unattended needs machinery that a chat agent can skip:
- Checkpointing: progress is saved to outputs/checkpoint.json after every task. If the run
  crashes, hits a rate limit or runs out of budget, running again resumes where it stopped.
- Retries with backoff: a failed task is retried (1 s, 2 s, ...) before being marked FAILED;
  one failed task doesn't kill the report.
- A budget: max LLM calls / tokens / seconds, enforced on every model call (tracing.Budget).
- Verification: the finished report is checked with the reflection grounding check, revised
  once if needed, and any number that still can't be traced is listed in the report.
- Human oversight at the right level: approve the PLAN once, instead of every step.

This reuses the earlier stages: specialists from multi_agent.py, ask_json from structured.py,
the grounding check from reflection.py, and Tracer/Budget from tracing.py.
"""
import json
import os
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from src.agent import MODEL, get_client
from src.charts import OUT_DIR
from src.multi_agent import run_specialist
from src.reflection import ungrounded_numbers
from src.structured import ask_json
from src.tracing import BudgetExceeded, TracedClient, Tracer


class ReportTask(BaseModel):
    id: str
    worker: Literal["analyst", "chart"]
    task: str = Field(min_length=3)


class ReportPlan(BaseModel):
    tasks: list[ReportTask] = Field(min_length=3, max_length=6)

    @model_validator(mode="after")
    def check_against_goal(self, info: ValidationInfo):
        # Seen live: the planner ignored "3 to 6 tasks" and "one chart" in the prompt.
        # Rules in validation are enforced; rules in a prompt are only requests.
        ids = [t.id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")  # results are stored by id
        goal = ((info.context or {}).get("goal") or "").lower()
        if any(w in goal for w in ("chart", "graph", "plot", "visual")) and not any(t.worker == "chart" for t in self.tasks):
            raise ValueError("the goal asks for a chart, but no task uses the chart worker")
        return self


PLANNER = (
    "You plan a short data report. Goal: {goal}\nCSV columns: {columns}\n"
    "Write 3 to 6 tasks for specialists: 'analyst' computes numbers with pandas tools, 'chart' saves one "
    "chart of a column aggregated by another. Each task must be self-contained (the specialist sees only "
    "its task text) and ask for exact numbers. Include at most 2 chart tasks.\n"
    'Reply in JSON: {{"tasks": [{{"id": "t1", "worker": "analyst", "task": "..."}}]}}')
WRITER = (
    "Write a short report in Markdown for this goal: {goal}\n"
    "Use ONLY the task results below. Every number must appear in the results; do not compute new ones. "
    "Start with '# ' and a title, then a one-paragraph summary, then the insights as a numbered list. "
    "Include each chart with ![description](path) using the exact path from the results. "
    "If a task FAILED, don't guess its result.")


def load_state(path, goal):
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("goal") == goal and state.get("status") != "done":
            return state
    return {"goal": goal, "status": "new", "tasks": [], "results": {}, "log": []}


def save_state(path, state):
    """Write to a temp file then rename: a crash mid-write can't leave a corrupt checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_autonomous(goal, tools, client=None, out_dir=OUT_DIR, budget=None, approve_plan=None,
                   approve=None, max_retries=2, sleep=time.sleep, verbose=True):
    """Work towards the goal. Returns the final state dict (status, results, report_path, totals)."""
    out = Path(out_dir)
    ckpt = out / "checkpoint.json"
    tracer = Tracer(budget=budget, out_dir=out / "traces")
    llm = TracedClient(client or get_client(), tracer)
    state = load_state(ckpt, goal)

    def log(msg):
        state["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")
        if verbose:
            print(f"AUTO   {msg}")

    try:
        if not state["tasks"]:
            columns = ", ".join(tools.describe_data()["columns"])
            plan, _ = ask_json(llm, [{"role": "system", "content": PLANNER.format(goal=goal, columns=columns)},
                                     {"role": "user", "content": "Write the plan."}], ReportPlan, verbose=verbose,
                               context={"goal": goal})
            if plan is None:
                raise RuntimeError("could not produce a valid plan")
            if approve_plan and not approve_plan(plan.tasks):
                state["status"] = "plan rejected"
                log("plan rejected by the user")
                return state
            state["tasks"], state["status"] = [t.model_dump() for t in plan.tasks], "running"
            log(f"planned {len(plan.tasks)} tasks")
            save_state(ckpt, state)
        elif state["results"]:
            log(f"resuming: {len(state['results'])}/{len(state['tasks'])} tasks already done")

        for t in state["tasks"]:
            if t["id"] in state["results"]:
                continue
            for attempt in range(max_retries + 1):
                try:
                    result = run_specialist(t["worker"], t["task"], tools, llm, verbose, approve)
                    break
                except BudgetExceeded:
                    raise
                except Exception as e:
                    log(f"{t['id']} attempt {attempt + 1} failed: {str(e)[:120]}")
                    if attempt == max_retries:
                        result = f"FAILED: {str(e)[:200]}"
                    else:
                        sleep(2 ** attempt)
            state["results"][t["id"]] = result
            log(f"{t['id']} done ({t['worker']})")
            save_state(ckpt, state)

        state["report_path"] = write_report(goal, state, llm, out, log).as_posix()
        state["status"] = "done"
    except BudgetExceeded as e:
        state["status"] = f"stopped: {e}. Run again to resume."
        log(state["status"])
    state["totals"] = tracer.totals()
    save_state(ckpt, state)
    return state


def write_report(goal, state, llm, out, log):
    results = "\n".join(f"[{t['id']} {t['worker']}] {t['task']}\nRESULT: {state['results'][t['id']]}\n"
                        for t in state["tasks"])
    messages = [{"role": "system", "content": WRITER.format(goal=goal)},
                {"role": "user", "content": f"Task results:\n{results}"}]
    report = llm.chat.completions.create(model=MODEL, messages=messages, max_tokens=1500).choices[0].message.content or ""
    if bad := ungrounded_numbers(report, [results], goal):
        log(f"report has untraceable numbers {bad}; asking for a revision")
        messages += [{"role": "assistant", "content": report},
                     {"role": "user", "content": f"These numbers are not in the task results: {bad}. "
                                                 "Remove or correct them and return the whole report."}]
        report = llm.chat.completions.create(model=MODEL, messages=messages, max_tokens=1500).choices[0].message.content or ""
        if still := ungrounded_numbers(report, [results], goal):
            report += ("\n\n---\n**Unverified:** these numbers could not be traced to a tool result: "
                       + ", ".join(f"{n:g}" for n in still))
    # Charts are saved as outputs/<name>.png; the report lives in outputs/ too, so link relatively.
    report = report.replace(f"]({Path(OUT_DIR).as_posix()}/", "](")
    path = out / "report.md"
    path.write_text(report, encoding="utf-8")
    log(f"report written to {path.as_posix()}")
    return path
