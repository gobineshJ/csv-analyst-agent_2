"""The same CSV agent rebuilt with LangGraph, for comparison with agent.py.

LangGraph supplies: the loop (graph edges), tool execution (ToolNode), message
history across questions (checkpointer), message trimming (trim_messages), and
pausing for a human (interrupt) with the state saved until the decision arrives.
"""
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from src.core.agent import MODEL, SYSTEM, TOOL_JSON_HINT, is_retryable
from src.core.schemas import RUN_PANDAS_DESCRIPTION
from src.paths import runtime_file
from src.patterns.hitl import apply_decision


def make_tools(csv, approve_code=False, charts=False):
    """Wrap the same CsvTools methods as LangChain tools. Docstrings become the descriptions."""

    @tool
    def describe_data() -> dict:
        """Get column names, types, row count and sample rows. Call this first."""
        return csv.describe_data()

    @tool
    def aggregate(column: str, how: Literal["sum", "mean", "count", "max", "min"],
                  group_by: str = "") -> dict:
        """Compute sum, mean, count, max or min of a column, optionally grouped by another column.
        Leave group_by empty for a single overall value."""
        return csv.aggregate(column, how, group_by or None)

    @tool
    def filter_rows(column: str, op: str, value: str) -> dict:
        """Return rows where a column matches a condition.
        Operators: ==, !=, >, <, >=, <=, contains (text only)."""
        return csv.filter_rows(column, op, value)

    @tool
    def top_n(column: str, n: int = 5, ascending: bool = False) -> dict:
        """Return the top N rows sorted by a column (descending by default)."""
        return csv.top_n(column, n, ascending)

    @tool(description=RUN_PANDAS_DESCRIPTION)
    def run_pandas(code: str) -> dict:
        if approve_code:
            # Pauses the whole graph here. The state is saved by the checkpointer, and
            # graph.stream(Command(resume=decision)) continues from this line, with
            # interrupt() returning the decision. (On resume the tool runs again from the top.)
            decision = interrupt({"tool": "run_pandas", "args": {"code": code}})
            args, rejected = apply_decision(decision, {"code": code})
            if rejected:
                return {"error": rejected}
            code = args["code"]
        return csv.run_pandas(code)

    @tool
    def make_chart(kind: Literal["bar", "line", "pie"], x: str, y: str,
                   how: Literal["sum", "mean", "count", "max", "min"] = "sum", title: str = "") -> dict:
        """Save a chart of column y aggregated by column x as a PNG. Returns the file path and the
        plotted numbers; describe the chart using those numbers. Use only when the user asks for a chart."""
        return csv.make_chart(kind, x, y, how, title)

    return [describe_data, aggregate, filter_rows, top_n, run_pandas] + ([make_chart] if charts else [])


def make_llm(callbacks=None):
    from langchain_groq import ChatGroq  # imported here so tests need no API key
    return ChatGroq(model=MODEL, temperature=0, callbacks=callbacks)


def sqlite_checkpointer(path=None):
    """Persistent memory: threads (and paused approvals) survive a restart."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver(sqlite3.connect(path or runtime_file("langgraph.db"), check_same_thread=False))


def build_graph(csv, llm=None, max_context_tokens=3000, approve_code=False, checkpointer=None, charts=False):
    """checkpointer: where thread state lives. Default MemorySaver (RAM, lost on exit).
    charts: also give the model make_chart (the web app does; the eval versions do not)."""
    tools = make_tools(csv, approve_code, charts)
    llm = llm or make_llm()
    model = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        # Memory: the checkpointer keeps the FULL history; the model only sees a trimmed view.
        # Trimming starts on a human message, so tool calls are never split from their results.
        view = trim_messages(state["messages"], strategy="last", max_tokens=max_context_tokens,
                             token_counter=count_tokens_approximately, start_on="human")
        if not view:  # the latest turn alone is over budget: send it anyway
            last_human = max(i for i, m in enumerate(state["messages"]) if m.type == "human")
            view = state["messages"][last_human:]
        messages = [SystemMessage(SYSTEM)] + view
        try:
            reply = model.invoke(messages)
        except Exception as e:  # broken tool-call JSON: same one retry as agent.chat()
            if not is_retryable(e):
                raise
            reply = model.invoke(messages + [HumanMessage(TOOL_JSON_HINT)])
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)  # -> "tools" if tool calls, else END
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_agent_langgraph(question, graph, thread_id="default", max_steps=8, verbose=True, ask=None):
    """ask(request) -> decision answers interrupts (see hitl.py). Without it, a paused
    graph is left waiting and the request is returned, so a caller can resume it later
    with resume_langgraph()."""
    return _run(graph, {"messages": [("user", question)]}, thread_id, max_steps, verbose, ask)


def resume_langgraph(graph, decision, thread_id="default", max_steps=8, verbose=True, ask=None):
    """Continue a graph that is paused at an interrupt, e.g. after a human decided in a UI."""
    return _run(graph, Command(resume=decision), thread_id, max_steps, verbose, ask)


def _run(graph, inputs, thread_id, max_steps, verbose, ask):
    answer, pending = _stream(graph, inputs, thread_id, max_steps, verbose)
    while pending is not None and ask is not None:
        answer, pending = _stream(graph, Command(resume=ask(pending)), thread_id, max_steps, verbose)
    return answer if pending is None else f"Waiting for approval: {pending}"


STEP_LIMIT = "Stopped: hit the step limit."


def stream_events(graph, inputs, thread_id="default", max_steps=8, callbacks=None):
    """Run the graph until it finishes or pauses, yielding events as they happen:
        {"type": "call", "name", "args"}   the model asked for a tool
        {"type": "result", "name", "content"}
        {"type": "answer", "text"}
        {"type": "approval", "request"}    paused at interrupt(); resume with Command(resume=...)
    The terminal version prints these (_stream); the web app sends them to the browser."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 2 * max_steps + 1,
              "callbacks": callbacks or []}
    try:
        for update in graph.stream(inputs, config, stream_mode="updates"):
            if "__interrupt__" in update:
                yield {"type": "approval", "request": update["__interrupt__"][0].value}
                continue
            for node_output in update.values():
                for m in (node_output or {}).get("messages", []):
                    if isinstance(m, AIMessage):
                        for tc in m.tool_calls:
                            yield {"type": "call", "name": tc["name"], "args": tc["args"]}
                        if not m.tool_calls:
                            yield {"type": "answer", "text": m.content}
                    elif isinstance(m, ToolMessage):
                        yield {"type": "result", "name": m.name, "content": str(m.content)}
    except GraphRecursionError:
        # The thread now ends with a tool call that never ran. Close it, or the next
        # question on this thread would send a tool call with no result to the API.
        last = graph.get_state(config).values["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            closing = [ToolMessage("Stopped: step limit reached before this tool ran.",
                                   tool_call_id=tc["id"]) for tc in last.tool_calls]
            graph.update_state(config, {"messages": closing + [AIMessage(STEP_LIMIT)]})
        yield {"type": "answer", "text": STEP_LIMIT}


def _stream(graph, inputs, thread_id, max_steps, verbose):
    """Run the graph until it finishes or pauses. Returns (answer, interrupt request or None)."""
    answer, pending = STEP_LIMIT, None
    for e in stream_events(graph, inputs, thread_id, max_steps):
        if e["type"] == "call" and verbose:
            print(f"CALL   {e['name']} {e['args']}")
        elif e["type"] == "result" and verbose:
            print(f"RESULT {e['content'][:200]}")
        elif e["type"] == "answer":
            answer = e["text"]
        elif e["type"] == "approval":
            pending = e["request"]
    return answer, pending
