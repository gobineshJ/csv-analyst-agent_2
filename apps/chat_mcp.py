"""The agent as an MCP client; the tools run in an MCP server subprocess.  python -m apps.chat_mcp [csv]"""
import asyncio
import sys
from dotenv import load_dotenv
load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.core.agent import llm_summarizer
from src.core.memory import ConversationMemory
from src.frameworks.agent_mcp import run_agent_mcp, to_openai_tools
from src.patterns.hitl import console_approver


async def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_sales.csv"
    # The agent starts the MCP server as a subprocess and talks to it over stdio.
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.frameworks.mcp_server", path])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools = to_openai_tools(mcp_tools)
            print("Connected to MCP server. Tools:", [t.name for t in mcp_tools])
            memory = ConversationMemory(summarize=llm_summarizer())
            while True:
                q = (await asyncio.to_thread(input, "\nAsk a question (or 'quit'): ")).strip()
                if q.lower() in ("quit", "exit", ""):
                    break
                print("\nANSWER:", await run_agent_mcp(q, session, tools, memory=memory,
                                                         approve=console_approver))


asyncio.run(main())
