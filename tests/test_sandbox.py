import pytest

from src.sandbox import check_code, child_env, run_code
from src.tools import CsvTools

P = "data/sample_sales.csv"


@pytest.mark.parametrize("code", [
    "import os\nresult = 1",
    "from pathlib import Path\nresult = 1",
    "result = df.__class__",
    "result = ().__class__.__bases__",
    "result = open('secret.txt').read()",
    "result = eval('1+1')",
    "result = getattr(df, 'to_csv')",
    "result = pd.read_csv('other.csv')",
    "result = df.to_csv('out.csv')",
    "result = df.query('sales > 1')",
    "result = '{0.__class__}'.format(df)",
    "result = pd.ExcelWriter('x.xlsx')",
    "x = 1",  # no result
])
def test_dangerous_or_incomplete_code_is_rejected_before_running(code):
    assert check_code(code) is not None


def test_runs_pandas_code_and_returns_json_friendly_results():
    assert run_code(P, "result = df[(df.category == 'Clothing') & (df.region == 'North')].sales.mean()") == {"result": 640.0}
    grouped = run_code(P, "result = df.groupby('region').sales.sum()")
    assert grouped["result"] == {"North": 3940, "South": 1600}


def test_runtime_errors_and_timeouts_come_back_as_errors():
    assert "KeyError" in run_code(P, "result = df['nope']")["error"]
    assert "timed out" in run_code(P, "while True:\n    pass\nresult = 1", timeout=3)["error"]


def test_child_process_gets_no_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "should-not-leak")
    assert "GROQ_API_KEY" not in child_env()


def test_csvtools_exposes_run_pandas():
    assert CsvTools(P).run_pandas("result = len(df)") == {"result": 8}
