"""Where generated files go: databases, charts, reports, traces.

Everything under runtime/ is git-ignored and safe to delete (you lose saved conversations,
long-term memory and old reports).
"""
from pathlib import Path

RUNTIME = Path("runtime")
OUTPUTS = RUNTIME / "outputs"   # charts, report.md, checkpoint.json
TRACES = RUNTIME / "traces"     # one .jsonl file per traced run


def runtime_file(name):
    """Path for a file in runtime/, creating the folder if needed (SQLite won't create it)."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    return str(RUNTIME / name)
