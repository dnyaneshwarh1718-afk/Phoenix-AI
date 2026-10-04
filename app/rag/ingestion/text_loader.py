from pathlib import Path

from app.rag.models import Document
from app.rag.document_identity import canonical_document_id
from app.rag.ingestion.base_loader import (
    BaseDocumentLoader,
)


class TextLoader(BaseDocumentLoader):

    supported_extensions = {
        ".txt",
        ".md",
        ".py",
        ".sql",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
    }

    def load(
        self,
        file_path: str,
    ) -> Document:

        path = Path(file_path)

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        document_id = canonical_document_id(path)

        return Document(
            document_id=document_id,
            source_path=str(path.resolve()),
            file_name=path.name,
            file_type=path.suffix.lower(),
            text=text,
        )