"""Chart tool for the chart specialist agent (multi_agent.py) and the report agent (autonomous.py)."""
import re
from pathlib import Path

import matplotlib

from src.paths import OUTPUTS

matplotlib.use("Agg")  # draw to files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = OUTPUTS.as_posix()  # runtime/outputs
KINDS = ("bar", "line", "pie")
HOWS = ("sum", "mean", "count", "max", "min")


def make_chart(df, kind, x, y, how="sum", title="", out_dir=OUT_DIR):
    """Aggregate y by x and save a chart as PNG. Returns the path and the plotted numbers."""
    for col in (x, y):
        if col not in df.columns:
            return {"error": f"unknown column '{col}'. Available: {list(df.columns)}"}
    if kind not in KINDS:
        return {"error": f"kind must be one of {KINDS}"}
    if how not in HOWS:
        return {"error": f"how must be one of {HOWS}"}
    data = df.groupby(x)[y].agg(how).round(2)
    if len(data) > 30:
        return {"error": f"'{x}' has {len(data)} values; pick a column with 30 or fewer"}
    title = title or f"{how} of {y} by {x}"
    fig, ax = plt.subplots(figsize=(7, 4))
    if kind == "pie":
        ax.pie(data.values, labels=[str(i) for i in data.index], autopct="%1.0f%%")
    else:
        data.plot(kind=kind, ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(f"{how} of {y}")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / (re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60] + ".png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    # Return the numbers too: the agent must describe the chart from data, not imagine it.
    return {"path": path.as_posix(), "data": {str(k): v.item() if hasattr(v, "item") else v for k, v in data.items()}}
