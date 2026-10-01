from pathlib import Path

from app.rag.indexing.document_indexer import (
    DocumentIndexer,
    IndexStatus,
)


class FakeLoader:

    def __init__(self):
        self.calls = 0

    def load(self, path):
        self.calls += 1

        return {
            "path": path,
            "content": "Phoenix AI test document",
        }


class FakeChunker:

    def __init__(self):
        self.calls = 0

    def chunk(self, document):
        self.calls += 1

        return [
            {
                "text": "Phoenix AI is a modular AI system."
            },
            {
                "text": "Phoenix uses Qdrant and BM25."
            },
        ]


class FakeEmbeddingEngine:

    def __init__(self):
        self.calls = 0

    def embed_documents(self, documents):
        self.calls += 1

        return [
            [0.1, 0.2],
            [0.3, 0.4],
        ]


class FakeVectorStore:

    def __init__(self):
        self.calls = 0
        self.documents = []

    def add_documents(
        self,
        embeddings,
        chunks,
        document_id,
    ):
        self.calls += 1

        self.documents.append(
            {
                "embeddings": embeddings,
                "chunks": chunks,
                "document_id": document_id,
            }
        )


class FakeBM25Store:

    def __init__(self):
        self.calls = 0
        self.documents = []

    def add_documents(
        self,
        chunks,
        document_id,
    ):
        self.calls += 1

        self.documents.append(
            {
                "chunks": chunks,
                "document_id": document_id,
            }
        )


class FakeRegistry:

    def __init__(self):
        self.indexed = set()
        self.register_calls = 0
        self.mark_indexed_calls = 0

    def is_indexed(self, path):
        return path in self.indexed

    def register(self, path, indexed=False):
        self.register_calls += 1

        if indexed:
            self.indexed.add(path)

    def mark_indexed(self, path):
        self.mark_indexed_calls += 1
        self.indexed.add(path)


def create_indexer():

    loader = FakeLoader()

    chunker = FakeChunker()

    embedding_engine = FakeEmbeddingEngine()

    vector_store = FakeVectorStore()

    bm25_store = FakeBM25Store()

    registry = FakeRegistry()

    indexer = DocumentIndexer(
        loader=loader,
        chunker=chunker,
        embedding_engine=embedding_engine,
        vector_store=vector_store,
        bm25_store=bm25_store,
        registry=registry,
    )

    return (
        indexer,
        loader,
        chunker,
        embedding_engine,
        vector_store,
        bm25_store,
        registry,
    )


def test_document_is_indexed(tmp_path):

    document = tmp_path / "test.pdf"

    document.write_text(
        "Phoenix AI test document",
        encoding="utf-8",
    )

    (
        indexer,
        loader,
        chunker,
        embedding_engine,
        vector_store,
        bm25_store,
        registry,
    ) = create_indexer()

    result = indexer.index(
        document,
        document_id="doc-123",
    )

    assert result.status == IndexStatus.INDEXED

    assert result.document_id == "doc-123"

    assert result.chunk_count == 2

    assert loader.calls == 1

    assert chunker.calls == 1

    assert embedding_engine.calls == 1

    assert vector_store.calls == 1

    assert bm25_store.calls == 1

    assert registry.mark_indexed_calls == 1

    assert registry.is_indexed(document) is True


def test_duplicate_document_is_not_reindexed(
    tmp_path,
):

    document = tmp_path / "test.pdf"

    document.write_text(
        "Phoenix AI test document",
        encoding="utf-8",
    )

    (
        indexer,
        loader,
        chunker,
        embedding_engine,
        vector_store,
        bm25_store,
        registry,
    ) = create_indexer()

    registry.indexed.add(document)

    result = indexer.index(
        document,
        document_id="doc-123",
    )

    assert (
        result.status
        == IndexStatus.ALREADY_INDEXED
    )

    assert loader.calls == 0

    assert chunker.calls == 0

    assert embedding_engine.calls == 0

    assert vector_store.calls == 0

    assert bm25_store.calls == 0


def test_missing_document_fails(tmp_path):

    document = tmp_path / "does_not_exist.pdf"

    (
        indexer,
        *_,
    ) = create_indexer()

    result = indexer.index(document)

    assert result.status == IndexStatus.FAILED

    assert "does not exist" in result.message


def test_document_id_is_generated_when_missing(
    tmp_path,
):

    document = tmp_path / "test.pdf"

    document.write_text(
        "Phoenix AI test document",
        encoding="utf-8",
    )

    (
        indexer,
        *_,
    ) = create_indexer()

    result = indexer.index(document)

    assert result.status == IndexStatus.INDEXED

    assert result.document_id is not None

    assert len(result.document_id) == 16


def test_registry_is_updated_after_indexing(
    tmp_path,
):

    document = tmp_path / "test.pdf"

    document.write_text(
        "Phoenix AI test document",
        encoding="utf-8",
    )

    (
        indexer,
        _,
        _,
        _,
        _,
        _,
        registry,
    ) = create_indexer()

    result = indexer.index(document)

    assert result.status == IndexStatus.INDEXED

    assert registry.is_indexed(document) is True

    assert registry.register_calls == 1

    assert registry.mark_indexed_calls == 1


def test_empty_chunks_fail(tmp_path):

    document = tmp_path / "empty.pdf"

    document.write_text(
        "Phoenix AI test document",
        encoding="utf-8",
    )

    (
        indexer,
        loader,
        chunker,
        embedding_engine,
        vector_store,
        bm25_store,
        registry,
    ) = create_indexer()

    chunker.chunk = lambda document: []

    result = indexer.index(document)

    assert result.status == IndexStatus.FAILED

    assert (
        result.message
        == "Document produced no chunks."
    )

    assert embedding_engine.calls == 0

    assert vector_store.calls == 0

    assert bm25_store.calls == 0