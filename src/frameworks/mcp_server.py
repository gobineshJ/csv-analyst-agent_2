"""MCP server: exposes the CSV tools over stdio so any MCP client can use them.

Run directly:  python -m src.frameworks.mcp_server data/sample_sales.csv
Never print() to stdout here - stdout carries the protocol messages.
"""
import os
import sys
from typing import Literal

from mcp.server.mcpserver import MCPServer

from src.core.schemas import RUN_PANDAS_DESCRIPTION
from src.core.tools import CsvTools

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CSV_PATH", "data/sample_sales.csv")
csv = CsvTools(path)
mcp = MCPServer("csv-analyst")


@mcp.tool()
def describe_data() -> dict:
    """Get column names, types, row count and sample rows. Call this first."""
    return csv.describe_data()


@mcp.tool()
def aggregate(column: str, how: Literal["sum", "mean", "count", "max", "min"],
              group_by: str = "") -> dict:
    """Compute sum, mean, count, max or min of a column, optionally grouped by another column.

    Leave group_by empty for a single overall value.
    """
    return csv.aggregate(column, how, group_by or None)


@mcp.tool()
def filter_rows(column: str, op: str, value: str) -> dict:
    """Return rows where a column matches a condition.

    Operators: ==, !=, >, <, >=, <=, contains (text only).
    """
    return csv.filter_rows(column, op, value)


@mcp.tool()
def top_n(column: str, n: int = 5, ascending: bool = False) -> dict:
    """Return the top N rows sorted by a column (descending by default)."""
    return csv.top_n(column, n, ascending)


@mcp.tool(description=RUN_PANDAS_DESCRIPTION)
def run_pandas(code: str) -> dict:
    # The server runs the code; approving it is the client's job (see agent_mcp.py).
    return csv.run_pandas(code)


if __name__ == "__main__":
    mcp.run(transport="stdio")
