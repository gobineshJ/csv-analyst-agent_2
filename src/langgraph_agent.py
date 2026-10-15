"""The same CSV agent rebuilt with LangGraph, for comparison with agent.py.

LangGraph supplies: the loop (graph edges), tool execution (ToolNode), message
history across questions (checkpointer), and message trimming (trim_messages).
"""
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent import MODEL, SYSTEM


def make_tools(csv):
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

    return [describe_data, aggregate, filter_rows, top_n]


def build_graph(csv, llm=None, max_context_tokens=3000):
    tools = make_tools(csv)
    if llm is None:
        from langchain_groq import ChatGroq  # imported here so tests need no API key
        llm = ChatGroq(model=MODEL, temperature=0)
    model = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        # Memory: the checkpointer keeps the FULL history; the model only sees a trimmed view.
        # Trimming starts on a human message, so tool calls are never split from their results.
        view = trim_messages(state["messages"], strategy="last", max_tokens=max_context_tokens,
                             token_counter=count_tokens_approximately, start_on="human")
        if not view:  # the latest turn alone is over budget: send it anyway
            last_human = max(i for i, m in enumerate(state["messages"]) if m.type == "human")
            view = state["messages"][last_human:]
        return {"messages": [model.invoke([SystemMessage(SYSTEM)] + view)]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)  # -> "tools" if tool calls, else END
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=MemorySaver())


def run_agent_langgraph(question, graph, thread_id="default", max_steps=8, verbose=True):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 2 * max_steps + 1}
    answer = "Stopped: hit the step limit."
    try:
        for update in graph.stream({"messages": [("user", question)]}, config, stream_mode="updates"):
            for node_output in update.values():
                for m in node_output["messages"]:
                    if isinstance(m, AIMessage):
                        for tc in m.tool_calls:
                            if verbose:
                                print(f"CALL   {tc['name']} {tc['args']}")
                        if not m.tool_calls:
                            answer = m.content
                    elif isinstance(m, ToolMessage) and verbose:
                        print(f"RESULT {str(m.content)[:200]}")
    except GraphRecursionError:
        # The thread now ends with a tool call that never ran. Close it, or the next
        # question on this thread would send a tool call with no result to the API.
        last = graph.get_state(config).values["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            closing = [ToolMessage("Stopped: step limit reached before this tool ran.",
                                   tool_call_id=tc["id"]) for tc in last.tool_calls]
            graph.update_state(config, {"messages": closing + [AIMessage(answer)]})
    return answer
