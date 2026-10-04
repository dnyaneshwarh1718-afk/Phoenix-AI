from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from app.rag.document_identity import canonical_document_id


# ============================================================
# INDEX STATUS
# ============================================================

class IndexStatus(str, Enum):
    INDEXED = "INDEXED"
    ALREADY_INDEXED = "ALREADY_INDEXED"
    REINDEXED = "REINDEXED"
    FAILED = "FAILED"


# ============================================================
# INDEX RESULT
# ============================================================

@dataclass
class IndexResult:
    status: IndexStatus
    path: Path
    document_id: Optional[str] = None
    chunk_count: int = 0
    message: str = ""
    error: Optional[str] = None


# ============================================================
# DOCUMENT INDEXER
# ============================================================

class DocumentIndexer:
    """
    Phoenix AI Document Indexer.

    Pipeline:

        File
          ↓
        DocumentManager / Loader
          ↓
        DocumentChunker
          ↓
        EmbeddingEngine
          ↓
        ┌───────────────┐
        │               │
        ↓               ↓
      Qdrant           BM25
        │               │
        └───────┬───────┘
                ↓
        Document Registry

    Responsibilities:

    1. Validate document
    2. Generate deterministic document ID
    3. Detect duplicate documents
    4. Load document
    5. Chunk document
    6. Attach document ID to chunks
    7. Generate embeddings
    8. Store vectors in Qdrant
    9. Store chunks in BM25
    10. Update document registry

    The implementation intentionally supports:

    REAL PHOENIX COMPONENTS
        DocumentManager.load_document()
        QdrantStore.upsert_chunks()
        BM25Store.add_chunks()

    TEST / LEGACY COMPONENTS
        loader.load()
        vector_store.add_documents()
        bm25_store.add_documents()
        simple registry fakes
    """

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        loader: Any = None,
        chunker: Any = None,
        embedding_engine: Any = None,
        vector_store: Any = None,
        bm25_store: Any = None,
        registry: Any = None,
        *,
        document_manager: Any = None,
        qdrant_store: Any = None,
    ) -> None:

        # ----------------------------------------------------
        # Loader / DocumentManager compatibility
        # ----------------------------------------------------

        self.document_manager = (
            document_manager
            if document_manager is not None
            else loader
        )

        # ----------------------------------------------------
        # Vector store / Qdrant compatibility
        # ----------------------------------------------------

        self.qdrant_store = (
            qdrant_store
            if qdrant_store is not None
            else vector_store
        )

        # Keep aliases for backward compatibility
        self.loader = self.document_manager
        self.vector_store = self.qdrant_store

        self.chunker = chunker
        self.embedding_engine = embedding_engine
        self.bm25_store = bm25_store
        self.registry = registry

        # ----------------------------------------------------
        # Dependency validation
        # ----------------------------------------------------

        missing = []

        if self.document_manager is None:
            missing.append("document_manager/loader")

        if self.chunker is None:
            missing.append("chunker")

        if self.embedding_engine is None:
            missing.append("embedding_engine")

        if self.qdrant_store is None:
            missing.append("qdrant_store/vector_store")

        if self.bm25_store is None:
            missing.append("bm25_store")

        if self.registry is None:
            missing.append("registry")

        if missing:
            raise ValueError(
                "DocumentIndexer missing dependencies: "
                + ", ".join(missing)
            )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def index(
        self,
        path: str | Path,
        document_id: Optional[str] = None,
        *,
        force_reindex: bool = False,
    ) -> IndexResult:

        path = Path(path)

        # ----------------------------------------------------
        # 1. Validate path
        # ----------------------------------------------------

        if not path.exists():
            return self._failure(
                path=path,
                document_id=document_id,
                message="Document does not exist.",
                error=f"File not found: {path}",
            )

        if not path.is_file():
            return self._failure(
                path=path,
                document_id=document_id,
                message="Path is not a file.",
                error=f"Not a file: {path}",
            )

        # ----------------------------------------------------
        # 2. Generate deterministic document ID
        # ----------------------------------------------------

        resolved_document_id = (
            document_id
            or self._generate_document_id(path)
        )

        # ----------------------------------------------------
        # 3. Check registry
        # ----------------------------------------------------

        try:
            registry_indexed = self._is_document_indexed(path)
            already_indexed = registry_indexed and not force_reindex

        except Exception as exc:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                message="Document registry lookup failed.",
                error=exc,
            )

        if already_indexed:

            record = self._get_registry_record(path)

            return IndexResult(
                status=IndexStatus.ALREADY_INDEXED,
                path=path,
                document_id=(
                    self._record_document_id(record)
                    or resolved_document_id
                ),
                chunk_count=(
                    self._record_chunk_count(record)
                ),
                message=(
                    f"Document is already indexed: {path}"
                ),
            )

        # ----------------------------------------------------
        # 4. Load document
        # ----------------------------------------------------

        try:
            document = self._load_document(path)

        except Exception as exc:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                message="Document loading failed.",
                error=exc,
            )

        # ----------------------------------------------------
        # 5. Chunk document
        # ----------------------------------------------------

        try:
            chunks = self._chunk_document(document)

        except Exception as exc:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                message="Document chunking failed.",
                error=exc,
            )

        # ----------------------------------------------------
        # 6. Validate chunks
        # ----------------------------------------------------

        if not chunks:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                message="Document produced no chunks.",
                error="No chunks generated.",
            )

        # ----------------------------------------------------
        # 7. Attach document ID
        # ----------------------------------------------------

        try:
            self._attach_document_id(
                chunks,
                resolved_document_id,
            )

        except Exception as exc:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                chunk_count=len(chunks),
                message="Chunk metadata preparation failed.",
                error=exc,
            )

        # ----------------------------------------------------
        # 8. Generate embeddings
        # ----------------------------------------------------

        try:
            embeddings = self._embed_chunks(chunks)

        except Exception as exc:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                chunk_count=len(chunks),
                message="Embedding generation failed.",
                error=exc,
            )

        # ----------------------------------------------------
        # 9. Validate embeddings
        # ----------------------------------------------------

        if not embeddings:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                chunk_count=len(chunks),
                message="Embedding engine returned no embeddings.",
                error="Empty embedding result.",
            )

        if len(embeddings) != len(chunks):
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                chunk_count=len(chunks),
                message="Embedding count does not match chunk count.",
                error=(
                    f"chunks={len(chunks)}, "
                    f"embeddings={len(embeddings)}"
                ),
            )

        # ----------------------------------------------------
        # 10. Index vectors
        # ----------------------------------------------------

        try:
            self._index_vectors(
                chunks=chunks,
                embeddings=embeddings,
                document_id=resolved_document_id,
            )

        except Exception as exc:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                chunk_count=len(chunks),
                message="Vector store indexing failed.",
                error=exc,
            )

        # ----------------------------------------------------
        # 10b. Remove stale vector chunks after successful reindex
        # ----------------------------------------------------

        if registry_indexed:
            try:
                cleanup = getattr(self.qdrant_store, "delete_document_except", None)
                if callable(cleanup):
                    cleanup(
                        resolved_document_id,
                        {str(getattr(chunk, "chunk_id", "")) for chunk in chunks},
                    )
            except Exception as exc:
                return self._failure(
                    path=path,
                    document_id=resolved_document_id,
                    chunk_count=len(chunks),
                    message="Stale vector cleanup failed.",
                    error=exc,
                )

        # ----------------------------------------------------
        # 11. Index BM25
        # ----------------------------------------------------

        try:
            self._index_bm25(
                chunks=chunks,
                document_id=resolved_document_id,
            )

        except Exception as exc:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                chunk_count=len(chunks),
                message="BM25 indexing failed.",
                error=exc,
            )

        # ----------------------------------------------------
        # 12. Update registry
        # ----------------------------------------------------

        try:
            self._update_registry(
                path=path,
                document_id=resolved_document_id,
                chunk_count=len(chunks),
            )

        except Exception as exc:
            return self._failure(
                path=path,
                document_id=resolved_document_id,
                chunk_count=len(chunks),
                message="Document registry update failed.",
                error=exc,
            )

        # ----------------------------------------------------
        # 13. Success
        # ----------------------------------------------------

        return IndexResult(
            status=IndexStatus.INDEXED,
            path=path,
            document_id=resolved_document_id,
            chunk_count=len(chunks),
            message=(
                f"Document indexed successfully: {path}"
            ),
        )

    def index_document(
        self,
        path: str | Path,
        document_id: Optional[str] = None,
        *,
        force_reindex: bool = False,
    ) -> IndexResult:
        """Index a document, optionally rebuilding stale/missing backend state.

        ``force_reindex`` is used when the registry says a document is indexed
        but Qdrant/BM25 no longer contain the corresponding chunks.
        """
        return self.index(path, document_id=document_id, force_reindex=force_reindex)

    def is_index_consistent(self, path: str | Path, document_id: str | None = None) -> bool:
        """Verify that registry, Qdrant and BM25 contain the same document.

        The registry is metadata only; it must never be treated as proof that
        the retrieval backends still contain the document.
        """
        path = Path(path)
        record = self._get_registry_record(path)
        doc_id = document_id or self._record_document_id(record)
        expected = self._record_chunk_count(record)
        if not doc_id or expected <= 0:
            return False

        qdrant_count = 0
        bm25_count = 0
        qdrant = self.qdrant_store
        if qdrant is not None:
            count_fn = getattr(qdrant, "document_chunk_count", None)
            if callable(count_fn):
                qdrant_count = int(count_fn(doc_id))
            else:
                has_fn = getattr(qdrant, "has_document", None)
                qdrant_count = expected if callable(has_fn) and has_fn(doc_id) else 0

        bm25 = self.bm25_store
        if bm25 is not None:
            count_fn = getattr(bm25, "document_chunk_count", None)
            if callable(count_fn):
                bm25_count = int(count_fn(doc_id))
            else:
                has_fn = getattr(bm25, "has_document", None)
                bm25_count = expected if callable(has_fn) and has_fn(doc_id) else 0

        return qdrant_count >= expected and bm25_count >= expected

    # ========================================================
    # REGISTRY
    # ========================================================

    def _is_document_indexed(
        self,
        path: Path,
    ) -> bool:

        registry = self.registry

        # ----------------------------------------------------
        # Preferred Phoenix registry API
        # ----------------------------------------------------

        method = getattr(
            registry,
            "is_indexed",
            None,
        )

        if callable(method):
            return bool(
                method(path)
            )

        # ----------------------------------------------------
        # contains() + get() style API
        # ----------------------------------------------------

        contains = getattr(
            registry,
            "contains",
            None,
        )

        if callable(contains):

            if not contains(path):
                return False

            get_method = getattr(
                registry,
                "get",
                None,
            )

            if callable(get_method):
                record = get_method(path)

                if record is not None:
                    status = getattr(
                        record,
                        "status",
                        None,
                    )

                    if isinstance(status, str):
                        return status.lower() == "indexed"

            return False

        # ----------------------------------------------------
        # Simple fake registry
        # ----------------------------------------------------

        indexed = getattr(
            registry,
            "indexed",
            None,
        )

        if indexed is not None:
            return (
                path in indexed
                or str(path) in indexed
            )

        return False

    def _get_registry_record(
        self,
        path: Path,
    ) -> Any:

        registry = self.registry

        # Current/simple registry
        method = getattr(
            registry,
            "get",
            None,
        )

        if callable(method):
            try:
                return method(path)
            except Exception:
                pass

        # Registry with find_by_path()
        method = getattr(
            registry,
            "find_by_path",
            None,
        )

        if callable(method):
            try:
                return method(str(path))
            except Exception:
                pass

        return None

    def _update_registry(
        self,
        path: Path,
        document_id: str,
        chunk_count: int,
    ) -> None:

        registry = self.registry

        # ----------------------------------------------------
        # Preferred simple/current registry API
        #
        # register(path, indexed=False)
        # mark_indexed(path)
        # ----------------------------------------------------

        register = getattr(
            registry,
            "register",
            None,
        )

        mark_indexed = getattr(
            registry,
            "mark_indexed",
            None,
        )

        if callable(register):

            # If mark_indexed exists, this is the
            # simple registry used by our unit tests.
            if callable(mark_indexed):

                try:
                    register(
                        path,
                        indexed=False,
                    )
                except TypeError:
                    register(
                        str(path),
                        indexed=False,
                    )

                mark_indexed(path)
                return

            # ------------------------------------------------
            # Full persistent DocumentRecord registry
            # ------------------------------------------------

            file_type = path.suffix.lower()

            register_kwargs = {
                "document_id": document_id,
                "file_name": path.name,
                "source_path": str(path),
                "file_type": file_type,
                "chunk_count": chunk_count,
                "status": "indexed",
            }

            try:
                signature = inspect.signature(
                    register
                )

                parameters = signature.parameters

                if all(
                    key in parameters
                    for key in register_kwargs
                ):
                    register(
                        **register_kwargs
                    )
                    return

            except (TypeError, ValueError):
                pass

            # Last compatibility attempt
            try:
                register(
                    path,
                    indexed=True,
                )
                return
            except TypeError:
                pass

        # ----------------------------------------------------
        # Registry with only mark_indexed()
        # ----------------------------------------------------

        if callable(mark_indexed):
            mark_indexed(path)
            return

        # ----------------------------------------------------
        # Very simple fake registry
        # ----------------------------------------------------

        indexed = getattr(
            registry,
            "indexed",
            None,
        )

        if indexed is not None:
            indexed.add(path)
            return

        raise AttributeError(
            "Registry does not provide a supported "
            "registration API."
        )

    @staticmethod
    def _record_document_id(
        record: Any,
    ) -> Optional[str]:

        if record is None:
            return None

        return getattr(
            record,
            "document_id",
            None,
        )

    @staticmethod
    def _record_chunk_count(
        record: Any,
    ) -> int:

        if record is None:
            return 0

        value = getattr(
            record,
            "chunk_count",
            0,
        )

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    # ========================================================
    # DOCUMENT LOADING
    # ========================================================

    def _load_document(
        self,
        path: Path,
    ) -> Any:

        manager = self.document_manager

        # ----------------------------------------------------
        # REAL PHOENIX API
        # ----------------------------------------------------

        method = getattr(
            manager,
            "load_document",
            None,
        )

        if callable(method):
            return method(
                str(path)
            )

        # ----------------------------------------------------
        # TEST / LEGACY API
        # ----------------------------------------------------

        method = getattr(
            manager,
            "load",
            None,
        )

        if callable(method):
            return method(path)

        raise AttributeError(
            "DocumentManager/Loader must provide "
            "load_document(path) or load(path)."
        )

    # ========================================================
    # CHUNKING
    # ========================================================

    def _chunk_document(
        self,
        document: Any,
    ) -> list[Any]:

        method = getattr(
            self.chunker,
            "chunk",
            None,
        )

        if not callable(method):
            raise AttributeError(
                "Chunker must provide chunk(document)."
            )

        chunks = method(document)

        if chunks is None:
            return []

        return list(chunks)

    # ========================================================
    # CHUNK METADATA
    # ========================================================

    @staticmethod
    def _attach_document_id(
        chunks: list[Any],
        document_id: str,
    ) -> None:

        for chunk in chunks:

            # ------------------------------------------------
            # Dictionary chunks
            # ------------------------------------------------

            if isinstance(chunk, dict):

                chunk["document_id"] = document_id
                metadata = chunk.get("metadata")
                if isinstance(metadata, dict):
                    metadata["document_id"] = document_id

                continue

            # ------------------------------------------------
            # Object chunks
            # ------------------------------------------------

            try:

                current = getattr(
                    chunk,
                    "document_id",
                    None,
                )

                if current != document_id:
                    setattr(chunk, "document_id", document_id)
                metadata = getattr(chunk, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["document_id"] = document_id

            except Exception:
                # Immutable objects are allowed.
                # Real DocumentChunk should already carry
                # document_id from the chunker.
                pass

    # ========================================================
    # TEXT EXTRACTION
    # ========================================================

    @staticmethod
    def _chunk_text(
        chunk: Any,
    ) -> str:

        # String
        if isinstance(chunk, str):
            return chunk

        # Dictionary
        if isinstance(chunk, dict):

            return str(
                chunk.get(
                    "text",
                    "",
                )
            )

        # Object
        text = getattr(
            chunk,
            "text",
            None,
        )

        if text is not None:
            return str(text)

        return str(chunk)

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def _embed_chunks(
        self,
        chunks: list[Any],
    ) -> list[Any]:

        engine = self.embedding_engine

        texts = [
            self._chunk_text(chunk)
            for chunk in chunks
        ]

        # ----------------------------------------------------
        # Standard Phoenix API
        # ----------------------------------------------------

        method = getattr(
            engine,
            "embed_documents",
            None,
        )

        if callable(method):
            return list(
                method(texts)
            )

        # ----------------------------------------------------
        # Compatibility APIs
        # ----------------------------------------------------

        method = getattr(
            engine,
            "embed",
            None,
        )

        if callable(method):
            return list(
                method(texts)
            )

        method = getattr(
            engine,
            "encode",
            None,
        )

        if callable(method):
            result = method(texts)

            if hasattr(
                result,
                "tolist",
            ):
                result = result.tolist()

            return list(result)

        raise AttributeError(
            "EmbeddingEngine must provide "
            "embed_documents(), embed(), or encode()."
        )

    # ========================================================
    # VECTOR STORE
    # ========================================================

    def _index_vectors(
        self,
        chunks: list[Any],
        embeddings: list[Any],
        document_id: str,
    ) -> None:

        store = self.qdrant_store

        # ----------------------------------------------------
        # REAL PHOENIX QDRANT
        # ----------------------------------------------------

        method = getattr(
            store,
            "upsert_chunks",
            None,
        )

        if callable(method):

            normalized_embeddings = []

            for vector in embeddings:

                if hasattr(
                    vector,
                    "tolist",
                ):
                    vector = vector.tolist()

                normalized_embeddings.append(
                    list(vector)
                )

            method(
                chunks=chunks,
                vectors=normalized_embeddings,
            )

            return

        # ----------------------------------------------------
        # TEST / LEGACY VECTOR STORE
        # ----------------------------------------------------

        method = getattr(
            store,
            "add_documents",
            None,
        )

        if callable(method):

            try:

                method(
                    embeddings,
                    chunks,
                    document_id,
                )

            except TypeError:

                try:

                    method(
                        embeddings=embeddings,
                        chunks=chunks,
                        document_id=document_id,
                    )

                except TypeError:

                    method(chunks)

            return

        # ----------------------------------------------------
        # Generic add()
        # ----------------------------------------------------

        method = getattr(
            store,
            "add",
            None,
        )

        if callable(method):

            try:
                method(
                    embeddings,
                    chunks,
                    document_id,
                )

            except TypeError:

                try:
                    method(
                        chunks=chunks,
                        embeddings=embeddings,
                        document_id=document_id,
                    )

                except TypeError:
                    method(chunks)

            return

        raise AttributeError(
            "Vector store must provide "
            "upsert_chunks(), add_documents(), "
            "or add()."
        )

    # ========================================================
    # BM25
    # ========================================================

    def _index_bm25(
        self,
        chunks: list[Any],
        document_id: str,
    ) -> None:

        store = self.bm25_store

        # ----------------------------------------------------
        # REAL PHOENIX BM25
        # ----------------------------------------------------

        method = getattr(
            store,
            "add_chunks",
            None,
        )

        if callable(method):

            try:

                method(
                    chunks,
                    document_id=document_id,
                )

            except TypeError:

                method(chunks)

            return

        # ----------------------------------------------------
        # TEST / LEGACY BM25
        # ----------------------------------------------------

        method = getattr(
            store,
            "add_documents",
            None,
        )

        if callable(method):

            try:

                method(
                    chunks,
                    document_id,
                )

            except TypeError:

                try:

                    method(
                        chunks=chunks,
                        document_id=document_id,
                    )

                except TypeError:

                    method(chunks)

            return

        # ----------------------------------------------------
        # Generic add()
        # ----------------------------------------------------

        method = getattr(
            store,
            "add",
            None,
        )

        if callable(method):

            try:

                method(
                    chunks,
                    document_id,
                )

            except TypeError:

                try:

                    method(
                        chunks=chunks,
                        document_id=document_id,
                    )

                except TypeError:

                    method(chunks)

            return

        raise AttributeError(
            "BM25 store must provide "
            "add_chunks(), add_documents(), "
            "or add()."
        )

    # ========================================================
    # DOCUMENT ID
    # ========================================================

    @staticmethod
    def _generate_document_id(
        path: Path,
    ) -> str:

        return canonical_document_id(path)

    # ========================================================
    # ERROR HELPERS
    # ========================================================

    @staticmethod
    def _format_error(
        error: Exception | str,
    ) -> str:

        if isinstance(
            error,
            Exception,
        ):

            return (
                f"{type(error).__name__}: "
                f"{error}"
            )

        return str(error)

    @classmethod
    def _failure(
        cls,
        *,
        path: Path,
        document_id: Optional[str],
        message: str,
        error: Exception | str,
        chunk_count: int = 0,
    ) -> IndexResult:

        return IndexResult(
            status=IndexStatus.FAILED,
            path=path,
            document_id=document_id,
            chunk_count=chunk_count,
            message=message,
            error=cls._format_error(error),
        )