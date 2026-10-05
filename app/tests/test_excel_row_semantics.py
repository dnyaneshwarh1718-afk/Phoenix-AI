from app.rag.ingestion.excel_loader import ExcelLoader
from openpyxl import Workbook


def test_excel_loader_preserves_product_and_revenue_on_same_row(tmp_path):
    path = tmp_path / "sales.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Product", "Revenue", "Embedding Dimension"])
    ws.append(["Engine A", 14400, 768])
    ws.append(["Engine B", 12000, 768])
    wb.save(path)

    doc = ExcelLoader().load(str(path))
    assert "Product: Engine A" in doc.text
    assert "Revenue: 14400" in doc.text
    assert "Embedding Dimension: 768" in doc.text
