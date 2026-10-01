
from app.rag.models import RetrievalResult


class CitationEngine:
    def build(self, results: list[RetrievalResult]) -> list[dict]:
        citations = []

        for index, result in enumerate(results, start=1):
            citations.append(
                {
                    "citation_id": index,
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "file_name": result.metadata.get("file_name"),
                    "source_path": result.metadata.get("source_path"),
                    "page": result.metadata.get("page"),
                    "slide": result.metadata.get("slide"),
                    "sheet": result.metadata.get("sheet"),
                    "score": result.score,
                }
            )

        return citations
