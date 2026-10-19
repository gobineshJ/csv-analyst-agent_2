"""Same agent loop as agent.py, but the tools come from an MCP server.

Differences from agent.py: tool schemas are discovered from the server (no schemas.py),
and tools are run with session.call_tool() instead of calling CsvTools directly.
"""
import asyncio
import json

from src.agent import SYSTEM, chat, get_client
from src.hitl import RISKY_TOOLS, apply_decision
from src.memory import ConversationMemory


def to_openai_tools(mcp_tools):
    """Convert MCP tool definitions to the format Groq/OpenAI expect."""
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description or "",
                          "parameters": t.input_schema}} for t in mcp_tools]


def result_text(result):
    return "".join(c.text for c in result.content if c.type == "text")


async def run_agent_mcp(question, session, tools, max_steps=8, verbose=True, memory=None, client=None,
                        approve=None):
    client = client or get_client()
    memory = memory or ConversationMemory()
    memory.add({"role": "user", "content": question})
    answer = "Stopped: hit the step limit."
    for _ in range(max_steps):
        resp = await asyncio.to_thread(chat, client, memory.build(SYSTEM), tools)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            break

        memory.add({"role": "assistant", "content": msg.content,
                    "tool_calls": [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name,
                                                 "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                if verbose:
                    print(f"CALL   {name} {args}  (via MCP)")
                rejected = None
                if approve and name in RISKY_TOOLS:  # the approver may block on input(), so run it off the loop
                    decision = await asyncio.to_thread(approve, {"tool": name, "args": args})
                    args, rejected = apply_decision(decision, args)
                text = (json.dumps({"error": rejected}) if rejected
                        else result_text(await session.call_tool(name, args)))
            except Exception as e:
                text = json.dumps({"error": str(e)})
            if verbose:
                print(f"RESULT {text[:200]}")
            memory.add({"role": "tool", "tool_call_id": tc.id, "content": text})
    memory.add({"role": "assistant", "content": answer})
    if memory.compact() and verbose:
        print("[memory] older turns summarized")
    return answer
