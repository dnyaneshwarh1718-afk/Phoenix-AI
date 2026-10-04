import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.rag.models import DocumentChunk


class QdrantStore:
    """
    Qdrant vector store for Phoenix AI.

    Responsibilities:
    - Create/verify the Phoenix collection
    - Convert Phoenix chunk IDs into valid Qdrant UUIDs
    - Upsert document chunks and embeddings
    - Perform dense vector search
    - Delete all chunks belonging to a document
    - Return payloads required by the retrieval layer

    Phoenix maintains its own deterministic chunk_id.
    Qdrant receives a deterministic UUID derived from that chunk_id.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "phoenix_documents",
        vector_size: int = 768,
    ) -> None:

        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
        )

        self._ensure_collection()

    # ============================================================
    # COLLECTION MANAGEMENT
    # ============================================================

    def _ensure_collection(self) -> None:
        """
        Create the collection if it does not already exist.

        Phoenix currently uses:
        - vector size: 768
        - distance: COSINE
        """

        collections = self.client.get_collections().collections

        exists = any(
            collection.name == self.collection_name
            for collection in collections
        )

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    # ============================================================
    # ID MANAGEMENT
    # ============================================================

    @staticmethod
    def _qdrant_point_id(chunk_id: str) -> str:
        """
        Convert Phoenix chunk ID into a deterministic UUID.

        Why?

        Phoenix chunk IDs are application-level IDs.

        Qdrant only accepts:
            - unsigned integers
            - UUIDs

        Therefore we derive a UUID from the Phoenix chunk ID.

        The same chunk_id will ALWAYS produce the same UUID.
        """

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"phoenix-ai:chunk:{chunk_id}",
            )
        )

    # ============================================================
    # UPSERT
    # ============================================================

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        """
        Insert or update document chunks in Qdrant.
        """

        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks must match "
                "number of vectors."
            )

        points: list[PointStruct] = []

        for chunk, vector in zip(chunks, vectors):

            if len(vector) != self.vector_size:
                raise ValueError(
                    f"Invalid vector dimension for chunk "
                    f"{chunk.chunk_id}. "
                    f"Expected {self.vector_size}, "
                    f"received {len(vector)}."
                )

            qdrant_id = self._qdrant_point_id(
                chunk.chunk_id
            )

            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }

            points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload=payload,
                )
            )

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    # ============================================================
    # DOCUMENT HEALTH
    # ============================================================

    def document_chunk_count(self, document_id: str) -> int:
        """Return the number of indexed Qdrant points for one document."""
        if not document_id:
            return 0
        response = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )
        return int(response.count)

    def has_document(self, document_id: str) -> bool:
        return self.document_chunk_count(document_id) > 0

    # ============================================================
    # DENSE SEARCH
    # ============================================================

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ):
        """
        Perform dense vector similarity search.

        Uses query_points(), which is compatible with
        current qdrant-client versions.
        """

        if not query_vector:
            raise ValueError(
                "query_vector cannot be empty."
            )

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Invalid query vector dimension. "
                f"Expected {self.vector_size}, "
                f"received {len(query_vector)}."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return response.points

    # ============================================================
    # DOCUMENT SEARCH
    # ============================================================

    def search_document(
        self,
        query_vector: list[float],
        document_id: str,
        limit: int = 5,
    ):
        """
        Search only inside a specific document.

        This will become useful later when Phoenix supports:

        "Ask this document only."
        """

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Invalid query vector dimension. "
                f"Expected {self.vector_size}, "
                f"received {len(query_vector)}."
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return response.points

    def delete_document_except(self, document_id: str, keep_chunk_ids: set[str]) -> None:
        """Delete stale chunks after a successful reindex."""
        if not document_id:
            return
        points = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )[0]
        stale_ids = []
        for point in points:
            chunk_id = str((point.payload or {}).get("chunk_id", ""))
            if chunk_id and chunk_id not in keep_chunk_ids:
                stale_ids.append(point.id)
        if stale_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=stale_ids,
                wait=True,
            )

    # ============================================================
    # DELETE DOCUMENT
    # ============================================================

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete every chunk belonging to a document.
        """

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            ),
            wait=True,
        )

    # ============================================================
    # COUNT
    # ============================================================

    def count(self) -> int:
        """
        Return total number of vectors in the collection.
        """

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    # ============================================================
    # COLLECTION INFO
    # ============================================================

    def collection_exists(self) -> bool:
        """
        Check whether the Phoenix collection exists.
        """

        collections = self.client.get_collections().collections

        return any(
            collection.name == self.collection_name
            for collection in collections
        )

    def collection_info(self) -> Any:
        """
        Return Qdrant collection information.

        Useful for debugging and health checks.
        """

        return self.client.get_collection(
            collection_name=self.collection_name
        )

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self) -> bool:
        """
        Basic Qdrant connectivity test.
        """

        try:
            self.client.get_collections()
            return True

        except Exception:
            return False