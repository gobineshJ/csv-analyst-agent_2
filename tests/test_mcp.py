"""These tests start the real MCP server as a subprocess and talk to it over stdio."""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agent_mcp import result_text, run_agent_mcp, to_openai_tools
from src.tools import CsvTools
from tests.fakes import FakeClient, call, text

PARAMS = StdioServerParameters(command=sys.executable,
                               args=["-m", "src.mcp_server", "data/sample_sales.csv"])


def with_session(fn):
    async def runner():
        async with stdio_client(PARAMS) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                return await fn(session)
    return asyncio.run(runner())


def test_server_lists_the_tools_with_enum():
    async def go(s):
        return (await s.list_tools()).tools
    tools = {t.name: t for t in with_session(go)}
    assert set(tools) == {"describe_data", "aggregate", "filter_rows", "top_n", "run_pandas"}
    assert tools["aggregate"].input_schema["properties"]["how"]["enum"] == ["sum", "mean", "count", "max", "min"]
    assert to_openai_tools(tools.values())[0]["type"] == "function"


def test_mcp_results_match_calling_the_tools_directly():
    direct = CsvTools("data/sample_sales.csv")
    cases = [("describe_data", {}, direct.describe_data()),
             ("aggregate", {"column": "sales", "how": "sum", "group_by": "category"},
              direct.aggregate("sales", "sum", "category")),
             ("aggregate", {"column": "sales", "how": "mean"}, direct.aggregate("sales", "mean")),
             ("filter_rows", {"column": "sales", "op": ">", "value": "900"},
              direct.filter_rows("sales", ">", "900")),
             ("top_n", {"column": "units", "n": 2}, direct.top_n("units", 2)),
             ("aggregate", {"column": "nope", "how": "sum"}, direct.aggregate("nope", "sum"))]

    async def go(s):
        return [json.loads(result_text(await s.call_tool(n, a))) for n, a, _ in cases]
    got = with_session(go)
    for (name, args, expected), actual in zip(cases, got):
        assert actual == json.loads(json.dumps(expected, default=str)), (name, args)


def test_agent_loop_runs_tool_through_mcp():
    client = FakeClient([call("aggregate", '{"column":"sales","how":"sum","group_by":"category"}', "id1"),
                         text("Electronics is highest.")])

    async def go(s):
        tools = to_openai_tools((await s.list_tools()).tools)
        return await run_agent_mcp("q", s, tools, client=client, verbose=False)
    assert with_session(go) == "Electronics is highest."
    fed_back = client.requests[1][-1]
    assert fed_back["tool_call_id"] == "id1" and "2760" in fed_back["content"]


def test_bad_arguments_and_unknown_tool_come_back_as_text():
    client = FakeClient([call("aggregate", '{"column":"sales"}', "a"),
                         call("rm_rf", "{}", "b"), text("done")])

    async def go(s):
        tools = to_openai_tools((await s.list_tools()).tools)
        return await run_agent_mcp("q", s, tools, client=client, verbose=False)
    assert with_session(go) == "done"
    assert "validation error" in client.requests[1][-1]["content"].lower() or "error" in client.requests[1][-1]["content"].lower()
    assert "Unknown tool" in client.requests[2][-1]["content"]
