"""Tool descriptions the model reads.

The model only knows what is written here, so names, descriptions and
parameter types directly affect which tool it picks and how it calls it.

ANTHROPIC_TOOLS is the base definition. TOOLS converts it to the
OpenAI-style format used by Groq (and many other providers).
"""

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
