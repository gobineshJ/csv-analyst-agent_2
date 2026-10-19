"""Run pandas code written by the model, with several layers of protection.

1. Static check: the code is parsed and rejected if it imports anything, touches
   dunder/private attributes, calls dangerous builtins, or uses pandas file I/O.
2. Restricted globals: only `df` (the CSV), `pd` and a small set of safe builtins exist.
3. Separate process: the code runs in a child Python with no API keys in its
   environment, in an empty temp directory, and is killed after a timeout.

This is defence in depth for a learning project, NOT a real security boundary.
Python can't be sandboxed reliably from inside Python. For untrusted users, run the
child in a container or VM with no network (e.g. Docker, gVisor, e2b, Modal).
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

TIMEOUT_SECONDS = 15  # includes the ~1-2 s it takes the child to import pandas
MAX_ROWS = 20
MAX_CHARS = 4000

BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
                 "vars", "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit",
                 "memoryview", "type", "object", "super"}
# Methods that read/write files, or that look attributes up from a string, which the AST check
# can't see inside: "{0.__class__}".format(df) reaches a dunder without an ast.Attribute node.
BLOCKED_ATTRS = {"eval", "query", "pipe", "format", "format_map", "to_csv", "to_excel", "to_pickle",
                 "to_parquet", "to_json", "to_sql", "to_hdf", "to_feather", "to_clipboard", "to_html",
                 "to_latex", "to_xml", "to_stata", "to_orc", "to_markdown", "io", "HDFStore",
                 "ExcelWriter", "ExcelFile"}
SAFE_BUILTINS = ["abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list",
                 "max", "min", "print", "range", "round", "set", "sorted", "str", "sum", "tuple",
                 "zip", "isinstance", "ValueError", "KeyError", "Exception"]


def check_code(code):
    """Return an error message if the code uses something not allowed, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed; `pd` and `df` are already available"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global/nonlocal are not allowed"
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BLOCKED_ATTRS
                                                or node.attr.startswith("read_")):
            return f"attribute '{node.attr}' is not allowed"
        if isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            return f"name '{node.id}' is not allowed"
    if "result" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}:
        return "assign the final answer to a variable called `result`"
    return None


# Runs in the child process. Reads {"path", "code"} from stdin, prints one JSON object.
_CHILD = r"""
import builtins, contextlib, io, json, sys
import pandas as pd
req = json.load(sys.stdin)
df = pd.read_csv(req["path"])
safe = {name: getattr(builtins, name) for name in req["builtins"]}
env = {"__builtins__": safe, "pd": pd, "df": df}
printed = io.StringIO()
try:
    with contextlib.redirect_stdout(printed):
        exec(req["code"], env)
    r = env.get("result")
    # to_json turns numpy numbers into plain JSON numbers and index labels into strings
    if isinstance(r, pd.DataFrame):
        out = {"rows": len(r), "result": json.loads(r.head(req["max_rows"]).reset_index().to_json(orient="records"))}
    elif isinstance(r, pd.Series):
        out = {"rows": len(r), "result": json.loads(r.head(req["max_rows"]).to_json())}
    elif hasattr(r, "item"):
        out = {"result": r.item()}
    else:
        out = {"result": r}
except Exception as e:
    out = {"error": f"{type(e).__name__}: {e}"}
if printed.getvalue():
    out["printed"] = printed.getvalue()
print(json.dumps(out, default=str))
"""


def child_env():
    """The parent's environment minus anything that looks like a secret."""
    secret = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    return {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in secret)}


def run_code(path, code, timeout=TIMEOUT_SECONDS):
    if err := check_code(code):
        return {"error": f"code rejected: {err}"}
    request = json.dumps({"path": os.path.abspath(path), "code": code,
                          "builtins": SAFE_BUILTINS, "max_rows": MAX_ROWS})
    with tempfile.TemporaryDirectory() as workdir:
        try:
            done = subprocess.run([sys.executable, "-I", "-c", _CHILD], input=request, text=True,
                                  capture_output=True, timeout=timeout, cwd=workdir, env=child_env())
        except subprocess.TimeoutExpired:
            return {"error": f"code timed out after {timeout} s"}
    if done.returncode != 0:
        return {"error": f"code crashed: {done.stderr.strip()[-500:]}"}
    text = done.stdout.strip()
    if len(text) > MAX_CHARS:
        return {"error": f"result too large ({len(text)} characters). Return fewer rows or a summary."}
    return json.loads(text)
