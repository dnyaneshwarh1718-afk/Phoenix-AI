from pathlib import Path
from openpyxl import Workbook
from app.rag.structured.excel_analytical_retriever import ExcelAnalyticalRetriever


def _book(tmp_path):
    path = tmp_path / "sales.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Product", "Revenue"])
    ws.append(["A", 10])
    ws.append(["B", 30])
    ws.append(["C", 20])
    wb.save(path)
    return path


def test_minimum_revenue(tmp_path):
    results = ExcelAnalyticalRetriever().search("which product has the lowest revenue?", str(_book(tmp_path)))
    assert "A" in results[0].text and "10" in results[0].text


def test_top_n_revenue(tmp_path):
    results = ExcelAnalyticalRetriever().search("top 2 products by revenue", str(_book(tmp_path)))
    assert len(results) == 2
    assert "B" in results[0].text and "C" in results[1].text
