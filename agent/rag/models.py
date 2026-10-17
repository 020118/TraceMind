"""RAG domain models and replaceable storage/embedding ports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    source_path: str
    title: str
    media_type: str
    content_hash: str
    chunk_count: int
    created_at: str = ""
    updated_at: str = ""
    vector_status: str = "PENDING"
    vector_error: str = ""
    vector_indexed_at: str = ""


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    id: str
    document_id: str
    ordinal: int
    text: str
    heading: str = ""
    start_line: int | None = None
    end_line: int | None = None
    page_number: int | None = None
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: ChunkRecord
    document: DocumentRecord
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0

    @property
    def citation(self) -> str:
        location = self.document.source_path
        if self.chunk.page_number is not None:
            location += f"#page={self.chunk.page_number}"
        elif self.chunk.start_line is not None:
            if self.chunk.end_line and self.chunk.end_line != self.chunk.start_line:
                location += f"#L{self.chunk.start_line}-L{self.chunk.end_line}"
            else:
                location += f"#L{self.chunk.start_line}"
        return f"[{self.document.title}]({location})"


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: str
    source_path: str
    chunk_count: int
    unchanged: bool = False
    semantic_indexed: bool = False
    vector_store_synced: bool = False
    vector_status: str = "LOCAL_ONLY"


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    chunk_id: str
    score: float


class VectorStore(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    async def initialize(self) -> bool: ...

    async def replace_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None: ...

    async def search(
        self, query_vector: list[float], *, limit: int
    ) -> list[VectorSearchResult]: ...

    async def close(self) -> None: ...


class EmbeddingProvider(Protocol):
    model_name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def close(self) -> None: ...


class RAGStore(Protocol):
    def initialize(self) -> None: ...

    def get_document(self, document_id: str) -> DocumentRecord | None: ...

    def replace_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None: ...

    def list_documents(self) -> list[DocumentRecord]: ...

    def list_chunks(self) -> list[tuple[ChunkRecord, DocumentRecord]]: ...

    def get_chunks_for_document(self, document_id: str) -> list[ChunkRecord]: ...

    def list_documents_for_vector_sync(self) -> list[DocumentRecord]: ...

    def update_vector_status(
        self, document_id: str, status: str, error: str = ""
    ) -> None: ...

    def record_retrieval_metric(self, **values: object) -> None: ...

    def retrieval_metrics_summary(self, limit: int = 500) -> dict: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class ParsedSection:
    text: str
    heading: str = ""
    start_line: int | None = None
    end_line: int | None = None
    page_number: int | None = None


@dataclass(slots=True)
class RankedCandidate:
    chunk: ChunkRecord
    document: DocumentRecord
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)

