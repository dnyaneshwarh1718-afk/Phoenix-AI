import hashlib
from pathlib import Path

from pypdf import PdfReader

from app.rag.models import Document
from app.rag.ingestion.base_loader import (
    BaseDocumentLoader,
)


class PDFLoader(BaseDocumentLoader):

    supported_extensions = {".pdf"}

    def load(
        self,
        file_path: str,
    ) -> Document:

        path = Path(file_path)

        reader = PdfReader(str(path))

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = page.extract_text() or ""

            pages.append(
                f"\n[Page {page_number}]\n{text}"
            )

        full_text = "\n".join(pages)

        document_id = hashlib.sha256(
            str(path.resolve()).encode()
        ).hexdigest()[:16]

        return Document(
            document_id=document_id,
            source_path=str(path.resolve()),
            file_name=path.name,
            file_type="pdf",
            text=full_text,
            metadata={
                "page_count": len(reader.pages),
            },
        )