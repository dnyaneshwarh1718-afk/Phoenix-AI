from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    document_id: str
    source_path: str
    file_name: str
    file_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    Common retrieval result interface.

    Dense and BM25 results should expose the same
    core fields so they can be combined by RRF.
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    retrieval_method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = -1


@dataclass
class DenseRetrievalResult:
    """
    Result returned by dense semantic retrieval.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_method: str = "dense"


@dataclass
class BM25RetrievalResult:
    """
    Result returned by BM25 keyword retrieval.
    """

    chunk: DocumentChunk
    score: float
    retrieval_method: str = "bm25"

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def chunk_index(self) -> int:
        return self.chunk.chunk_index

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata