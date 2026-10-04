import hashlib
from pathlib import Path

import pandas as pd

from app.rag.models import Document
from app.rag.ingestion.base_loader import (
    BaseDocumentLoader,
)


class ExcelLoader(BaseDocumentLoader):

    supported_extensions = {
        ".xlsx",
        ".xls",
    }

    def load(
        self,
        file_path: str,
    ) -> Document:

        path = Path(file_path)

        sheets = pd.read_excel(
            path,
            sheet_name=None,
        )

        sections = []

        for sheet_name, dataframe in sheets.items():

            dataframe = dataframe.fillna("")

            csv_text = dataframe.to_csv(
                index=False
            )

            sections.append(
                f"\n[Sheet: {sheet_name}]\n"
                f"{csv_text}"
            )

        text = "\n".join(sections)

        document_id = hashlib.sha256(
            str(path.resolve()).encode()
        ).hexdigest()[:16]

        return Document(
            document_id=document_id,
            source_path=str(path.resolve()),
            file_name=path.name,
            file_type="excel",
            text=text,
            metadata={
                "sheet_count": len(sheets),
                "sheet_names": list(
                    sheets.keys()
                ),
            },
        )