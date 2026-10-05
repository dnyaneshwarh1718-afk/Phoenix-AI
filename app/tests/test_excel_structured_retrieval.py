from pathlib import Path

from openpyxl import Workbook

from app.rag.structured.excel_analytical_retriever import ExcelAnalyticalRetriever


def test_highest_revenue_returns_grounded_product(tmp_path: Path):
    path = tmp_path / "sales.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Product", "Revenue", "Embedding Dimension"])
    ws.append(["Engine A", 14400, 768])
    ws.append(["Engine B", 12000, 768])
    wb.save(path)

    results = ExcelAnalyticalRetriever().search(
        "According to the spreadsheet, which product has the highest revenue?",
        str(path),
    )

    assert results
    assert results[0].retrieval_method == "structured_excel"
    assert "Engine A" in results[0].text
    assert "14400" in results[0].text
    assert results[0].metadata["structured_operation"] == "max"
