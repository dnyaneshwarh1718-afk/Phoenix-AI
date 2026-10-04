from pathlib import Path

from app.rag.ingestion.base_loader import (
    BaseDocumentLoader,
)

from app.rag.ingestion.pdf_loader import (
    PDFLoader,
)

from app.rag.ingestion.docx_loader import (
    DOCXLoader,
)

from app.rag.ingestion.pptx_loader import (
    PPTXLoader,
)

from app.rag.ingestion.excel_loader import (
    ExcelLoader,
)

from app.rag.ingestion.text_loader import (
    TextLoader,
)


class LoaderFactory:

    def __init__(self):

        self.loaders: list[
            BaseDocumentLoader
        ] = [

            PDFLoader(),
            DOCXLoader(),
            PPTXLoader(),
            ExcelLoader(),
            TextLoader(),
        ]

    def get_loader(
        self,
        file_path: str,
    ) -> BaseDocumentLoader:

        extension = (
            Path(file_path)
            .suffix
            .lower()
        )

        for loader in self.loaders:

            if loader.supports(
                extension
            ):

                return loader

        raise ValueError(
            f"Unsupported file type: "
            f"{extension}"
        )