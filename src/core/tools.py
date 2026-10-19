import pandas as pd

from src.core.charts import make_chart
from src.core.sandbox import run_code
from src.patterns.guardrails import sanitize

MAX_ROWS = 20  # keep tool output small so it doesn't flood the context


class CsvTools:
    def __init__(self, path, guard=True):
        self.path = path
        self.df = pd.read_csv(path)
        self.guard = guard  # False lets you watch a prompt injection reach the model

    def _out(self, result):
        return sanitize(result) if self.guard else result

    def _check(self, *cols):
        for c in cols:
            if c and c not in self.df.columns:
                return {"error": f"unknown column '{c}'. Available: {list(self.df.columns)}"}
        return None

    def describe_data(self):
        return self._out({
            "rows": len(self.df),
            "columns": {c: str(t) for c, t in self.df.dtypes.items()},
            "sample": self.df.head(3).to_dict("records"),
        })

    def aggregate(self, column, how, group_by=None):
        if err := self._check(column, group_by):
            return err
        if how not in ("sum", "mean", "count", "max", "min"):
            return {"error": f"unsupported aggregation '{how}'"}
        if how != "count" and not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"error": f"'{column}' is not numeric, cannot compute {how}"}
        if group_by:
            return self._out(self.df.groupby(group_by)[column].agg(how).round(2).to_dict())
        return {"result": round(float(self.df[column].agg(how)), 2)}

    def filter_rows(self, column, op, value):
        if err := self._check(column):
            return err
        s = self.df[column]
        if pd.api.types.is_numeric_dtype(s):
            try:
                value = float(value)
            except ValueError:
                return {"error": f"'{column}' is numeric, '{value}' is not a number"}
        ops = {"==": s == value, "!=": s != value}
        if pd.api.types.is_numeric_dtype(s):
            ops.update({">": s > value, "<": s < value, ">=": s >= value, "<=": s <= value})
        else:
            ops["contains"] = s.astype(str).str.contains(str(value), case=False)
        if op not in ops:
            return {"error": f"operator '{op}' not valid here. Use one of {list(ops)}"}
        out = self.df[ops[op]]
        return self._out({"matches": len(out), "rows": out.head(MAX_ROWS).to_dict("records")})

    def top_n(self, column, n=5, ascending=False):
        if err := self._check(column):
            return err
        out = self.df.sort_values(column, ascending=ascending).head(min(int(n), MAX_ROWS))
        return self._out({"rows": out.to_dict("records")})

    def run_pandas(self, code):
        """Run model-written pandas code in the sandbox. The code must assign `result`."""
        return self._out(run_code(self.path, code))

    def make_chart(self, kind, x, y, how="sum", title=""):
        """Save a bar/line/pie chart of y aggregated by x. Used by the chart specialist."""
        return self._out(make_chart(self.df, kind, x, y, how, title))
