"""Tool descriptions the model reads.

The model only knows what is written here, so names, descriptions and
parameter types directly affect which tool it picks and how it calls it.

ANTHROPIC_TOOLS is the base definition. TOOLS converts it to the
OpenAI-style format used by Groq (and many other providers).
FIXED_TOOLS and CODE_TOOLS are subsets, used to compare the two styles in the eval.
"""

RUN_PANDAS_DESCRIPTION = (
    "Run Python/pandas code on the CSV for questions the other tools cannot answer "
    "(several filters, dates, derived values, percentages). The data is in `df` and pandas is `pd`. "
    "No imports, no file access. Assign the final answer to `result` (a number, string, "
    "Series or DataFrame); at most 20 rows are returned. Example: "
    "result = df[df.region == 'North'].groupby('category').sales.mean()"
)

ANTHROPIC_TOOLS = [
    {
        "name": "describe_data",
        "description": "Get column names, types, row count and sample rows. Call this first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "aggregate",
        "description": (
            "Compute sum, mean, count, max or min of a column, "
            "optionally grouped by another column."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "how": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "max", "min"],
                },
                "group_by": {"type": "string"},
            },
            "required": ["column", "how"],
        },
    },
    {
        "name": "filter_rows",
        "description": (
            "Return rows where a column matches a condition. "
            "Operators: ==, !=, >, <, >=, <=, contains (text only)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "op": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["column", "op", "value"],
        },
    },
    {
        "name": "top_n",
        "description": "Return the top N rows sorted by a column (descending by default).",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "n": {"type": "integer"},
                "ascending": {"type": "boolean"},
            },
            "required": ["column"],
        },
    },
    {
        "name": "run_pandas",
        "description": RUN_PANDAS_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "pandas code that sets `result`"}},
            "required": ["code"],
        },
    },
]

# OpenAI / Groq format: same content, different wrapper.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in ANTHROPIC_TOOLS
]
FIXED_TOOLS = [t for t in TOOLS if t["function"]["name"] != "run_pandas"]
CODE_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("describe_data", "run_pandas")]

# Only given to the chart specialist (multi_agent.py), so the other versions are unchanged.
MAKE_CHART = {
    "type": "function",
    "function": {
        "name": "make_chart",
        "description": ("Save a chart of one column aggregated by another as a PNG. Returns the file path "
                        "and the plotted numbers; describe the chart using those numbers."),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["bar", "line", "pie"]},
                "x": {"type": "string", "description": "column to group by (the categories)"},
                "y": {"type": "string", "description": "numeric column to aggregate"},
                "how": {"type": "string", "enum": ["sum", "mean", "count", "max", "min"]},
                "title": {"type": "string"},
            },
            "required": ["kind", "x", "y"],
        },
    },
}
CHART_TOOLS = [TOOLS[0], MAKE_CHART]  # describe_data + make_chart
