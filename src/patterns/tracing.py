"""Observability: record every LLM call and tool call with timing, tokens and cost.

Nothing in the agent code changes. Tracing is added from the outside:
- TracedClient wraps the Groq client (plain loop, MCP, reflection, structured, multi-agent).
- TracedTools wraps CsvTools (every version, including LangGraph, calls tools through it).
- TracingCallback plugs into LangChain/LangGraph models through the callback system.

Every event is one JSON line in runtime/traces/<run_id>.jsonl, so a run can be inspected afterwards:
    python -m src.patterns.tracing runtime/traces/<run_id>.jsonl

The same wrapper enforces a Budget (max LLM calls / tokens / seconds), which is what keeps
an autonomous agent from running up a bill (see autonomous.py).

Hosted tools (Langfuse, LangSmith, Arize Phoenix) do the same thing with a UI on top; to try
one, swap TracingCallback for their LangChain callback handler.
"""
import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

from langchain_core.callbacks import BaseCallbackHandler

from src.paths import TRACES

# US$ per 1M tokens (input, output). Placeholder values: check Groq's pricing page and update.
PRICES = {"openai/gpt-oss-120b": (0.15, 0.75)}


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, max_llm_calls=None, max_tokens=None, max_seconds=None):
        self.max_llm_calls, self.max_tokens, self.max_seconds = max_llm_calls, max_tokens, max_seconds

    def check(self, tracer):
        t = tracer.totals()
        if self.max_llm_calls is not None and t["llm_calls"] >= self.max_llm_calls:
            raise BudgetExceeded(f"LLM call budget used up ({self.max_llm_calls} calls)")
        if self.max_tokens is not None and t["input_tokens"] + t["output_tokens"] >= self.max_tokens:
            raise BudgetExceeded(f"token budget used up ({self.max_tokens} tokens)")
        if self.max_seconds is not None and time.time() - tracer.started >= self.max_seconds:
            raise BudgetExceeded(f"time budget used up ({self.max_seconds} s)")


class Tracer:
    def __init__(self, run_id=None, out_dir=TRACES, budget=None, write=True):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.events, self.started, self.budget = [], time.time(), budget
        self.path = Path(out_dir) / f"{self.run_id}.jsonl" if write else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, kind, **fields):
        event = {"t": round(time.time() - self.started, 3), "kind": kind, **fields}
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")

    def before_llm_call(self):
        if self.budget:
            self.budget.check(self)

    def totals(self):
        llm = [e for e in self.events if e["kind"] == "llm"]
        tools = [e for e in self.events if e["kind"] == "tool"]
        cost = 0.0
        for e in llm:
            price_in, price_out = PRICES.get(e.get("model"), (0, 0))
            cost += (e["input_tokens"] * price_in + e["output_tokens"] * price_out) / 1e6
        return {"llm_calls": len(llm), "tool_calls": len(tools),
                "tool_errors": sum(1 for e in tools if e.get("error")),
                "input_tokens": sum(e["input_tokens"] for e in llm),
                "output_tokens": sum(e["output_tokens"] for e in llm),
                "llm_seconds": round(sum(e["seconds"] for e in llm), 2),
                "tool_seconds": round(sum(e["seconds"] for e in tools), 2),
                "cost_usd": round(cost, 6)}


class TracedClient:
    """Drop-in for the Groq client: client.chat.completions.create(...) is timed and recorded."""

    def __init__(self, client, tracer):
        self.inner, self.tracer = client, tracer
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.tracer.before_llm_call()
        start = time.time()
        try:
            resp = self.inner.chat.completions.create(**kw)
        except Exception as e:
            self.tracer.record("llm", model=kw.get("model"), seconds=round(time.time() - start, 3),
                               input_tokens=0, output_tokens=0, error=str(e)[:300])
            raise
        usage = getattr(resp, "usage", None)
        msg = resp.choices[0].message
        self.tracer.record("llm", model=kw.get("model"), seconds=round(time.time() - start, 3),
                           input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                           output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                           tool_calls=[tc.function.name for tc in (msg.tool_calls or [])])
        return resp


class TracedTools:
    """Wraps CsvTools: every public method call is timed and recorded."""

    def __init__(self, tools, tracer):
        self._tools, self._tracer = tools, tracer

    def __getattr__(self, name):
        attr = getattr(self._tools, name)
        if name.startswith("_") or not callable(attr):
            return attr

        def traced(*args, **kwargs):
            start = time.time()
            out = attr(*args, **kwargs)
            self._tracer.record("tool", name=name, args=kwargs or list(args), seconds=round(time.time() - start, 3),
                                error=out.get("error") if isinstance(out, dict) else None)
            return out
        return traced


class TracingCallback(BaseCallbackHandler):
    """LangChain callback: records each chat-model call (LangGraph and the planner)."""

    def __init__(self, tracer, model=None):
        self.tracer, self.model, self.starts = tracer, model, {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self.tracer.before_llm_call()
        self.starts[run_id] = time.time()

    def on_llm_end(self, response, *, run_id, **kw):
        msg = response.generations[0][0].message
        usage = getattr(msg, "usage_metadata", None) or {}
        self.tracer.record("llm", model=self.model, seconds=round(time.time() - self.starts.pop(run_id, time.time()), 3),
                           input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                           tool_calls=[tc["name"] for tc in getattr(msg, "tool_calls", [])])


def summarize(path):
    """Print a readable timeline and totals for a trace file."""
    tracer = Tracer(write=False)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        tracer.events.append(e)
        if e["kind"] == "llm":
            what = f"LLM  {e['input_tokens']:>6} in {e['output_tokens']:>5} out"
            what += f"  -> {', '.join(e.get('tool_calls') or []) or 'answer'}" if not e.get("error") else f"  ERROR {e['error'][:60]}"
        else:
            what = f"TOOL {e['name']}" + (f"  ERROR {str(e['error'])[:60]}" if e.get("error") else "")
        print(f"{e['t']:>8.2f}s  {e['seconds']:>6.2f}s  {what}")
    print(json.dumps(tracer.totals(), indent=2))


if __name__ == "__main__":
    summarize(sys.argv[1])
