from dataclasses import dataclass, field
from typing import Any

from app.rag.vector_store.qdrant_store import QdrantStore
from app.rag.embeddings.embedding_engine import EmbeddingEngine


@dataclass
class DenseRetrievalResult:
    """
    Standard result returned by dense retrieval.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any] = field(
        default_factory=dict
    )
    retrieval_method: str = "dense"


class DenseRetriever:
    """
    Dense semantic retriever for Phoenix AI.

    Supports:

        Global Retrieval
        ----------------
        Query
          ↓
        Embedding
          ↓
        Qdrant
          ↓
        Dense Results


        Document-Specific Retrieval
        ---------------------------
        Query
          ↓
        Embedding
          ↓
        Qdrant
          ↓
        document_id filter
          ↓
        Dense Results
    """

    def __init__(
        self,
        vector_store: QdrantStore,
        embedding_engine: EmbeddingEngine,
    ) -> None:

        self.vector_store = vector_store
        self.embedding_engine = embedding_engine

    # ==========================================================
    # GLOBAL SEARCH
    # ==========================================================

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[DenseRetrievalResult]:
        """
        Search across all indexed documents.
        """

        if not query or not query.strip():
            return []

        if limit <= 0:
            return []

        query = query.strip()

        # ------------------------------------------------------
        # Generate query embedding
        # ------------------------------------------------------

        vector = self.embedding_engine.embed_query(
            query
        )

        # ------------------------------------------------------
        # Global Qdrant search
        # ------------------------------------------------------

        points = self.vector_store.search(
            vector,
            limit=limit,
        )

        return self._convert_points(
            points
        )

    # ==========================================================
    # DOCUMENT-SPECIFIC SEARCH
    # ==========================================================

    def search_document(
        self,
        query: str,
        document_id: str,
        limit: int = 5,
    ) -> list[DenseRetrievalResult]:
        """
        Search only inside a specific document.

        Parameters
        ----------
        query:
            User question.

        document_id:
            Phoenix document ID stored in the
            document registry and Qdrant.

        limit:
            Maximum number of results.

        Returns
        -------
        list[DenseRetrievalResult]

        Only chunks belonging to the specified
        document are returned.
        """

        if not query or not query.strip():
            return []

        if not document_id or not document_id.strip():
            return []

        if limit <= 0:
            return []

        query = query.strip()
        document_id = document_id.strip()

        # ------------------------------------------------------
        # Generate query embedding
        # ------------------------------------------------------

        vector = self.embedding_engine.embed_query(
            query
        )

        # ------------------------------------------------------
        # Document-specific Qdrant search
        # ------------------------------------------------------

        points = self.vector_store.search_document(
            query_vector=vector,
            document_id=document_id,
            limit=limit,
        )

        return self._convert_points(
            points
        )

    # ==========================================================
    # RESULT CONVERSION
    # ==========================================================

    @staticmethod
    def _convert_points(
        points,
    ) -> list[DenseRetrievalResult]:
        """
        Convert Qdrant points into Phoenix
        DenseRetrievalResult objects.

        Keeps global and document-specific
        retrieval behavior consistent.
        """

        results: list[
            DenseRetrievalResult
        ] = []

        for point in points:

            payload = point.payload or {}

            metadata = payload.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            chunk_id = payload.get(
                "chunk_id",
                str(point.id),
            )

            document_id = payload.get(
                "document_id",
                metadata.get(
                    "document_id",
                    "",
                ),
            )

            chunk_index = payload.get(
                "chunk_index",
                metadata.get(
                    "chunk_index",
                    -1,
                ),
            )

            text = payload.get(
                "text",
                "",
            )

            results.append(
                DenseRetrievalResult(
                    chunk_id=str(
                        chunk_id
                    ),
                    document_id=str(
                        document_id
                    ),
                    chunk_index=int(
                        chunk_index
                    ),
                    text=str(
                        text
                    ),
                    score=float(
                        point.score
                    ),
                    metadata=metadata,
                    retrieval_method="dense",
                )
            )

        return results