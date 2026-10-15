from src.tools import CsvTools

t = CsvTools("data/sample_sales.csv")


def test_sum_by_category():
    assert t.aggregate("sales", "sum", "category")["Electronics"] == 2760


def test_mean_by_category():
    assert t.aggregate("sales", "mean", "category")["Electronics"] == 920


def test_unknown_column_returns_error():
    assert "error" in t.aggregate("nope", "sum")


def test_mean_on_text_column_returns_error():
    assert "error" in t.aggregate("region", "mean")


def test_filter_numeric():
    assert t.filter_rows("sales", ">", "900")["matches"] == 2


def test_top_n():
    assert t.top_n("sales", 1)["rows"][0]["sales"] == 1200
