"""Hybrid retrieval for the Markdown based V2 memory store."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from agent.memory.store2 import MemoryStoreV2

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]

_SOURCE_LABELS = {
    "long_term": "长期记忆",
    "pending": "待整理事实",
    "recent_context": "近期上下文",
}
_SOURCE_PRIORITY = {
    "long_term": 1.0,
    "pending": 0.98,
    "recent_context": 0.92,
}
_LATIN_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


@dataclass(frozen=True, slots=True)
class MemoryChunk:
    source: str
    text: str
    search_text: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class MemorySearchHit:
    source: str
    text: str
    score: float
    semantic_score: float
    keyword_score: float


class HybridMemoryRetriever:
    """Retrieve relevant Markdown chunks with semantic + keyword scoring.

    Candidate embeddings are cached in SQLite. If the embedding provider is
    unavailable, retrieval continues with the deterministic keyword scorer.
    """

    def __init__(
        self,
        store: MemoryStoreV2,
        *,
        workspace: Path,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        top_k: int = 6,
        score_threshold: float = 0.35,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        embedder: Embedder | None = None,
    ) -> None:
        self._store = store
        self._model = model.strip()
        self._cache_namespace = f"{base_url.strip()}|{self._model or 'custom-embedder'}"
        self._top_k = max(1, min(int(top_k), 20))
        self._score_threshold = min(max(float(score_threshold), 0.0), 1.0)
        total_weight = max(float(semantic_weight) + float(keyword_weight), 0.001)
        self._semantic_weight = float(semantic_weight) / total_weight
        self._keyword_weight = float(keyword_weight) / total_weight
        self._embedder = embedder
        self._client: AsyncOpenAI | None = None
        if embedder is None and api_key.strip() and self._model:
            client_options: dict[str, str] = {"api_key": api_key}
            if base_url.strip():
                client_options["base_url"] = base_url.strip()
            self._client = AsyncOpenAI(**client_options)

        cache_dir = Path(workspace) / "memory"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_dir / "retrieval_cache.db"
        self._initialize_cache()

    @property
    def semantic_enabled(self) -> bool:
        return self._embedder is not None or self._client is not None

    async def build_context(self, query: str) -> str:
        hits = await self.search(query)
        if not hits:
            return ""
        lines = ["以下是根据当前问题检索到的用户记忆，仅作为参考："]
        for hit in hits:
            label = _SOURCE_LABELS.get(hit.source, hit.source)
            lines.append(f"- [{label}] {hit.text}")
        return "\n".join(lines)

    async def search(self, query: str) -> list[MemorySearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        chunks = self._collect_chunks()
        if not chunks:
            return []

        keyword_scores = [
            _keyword_similarity(query, chunk.search_text) for chunk in chunks
        ]
        semantic_scores: list[float] | None = None
        if self.semantic_enabled:
            try:
                vectors = await self._get_embeddings(
                    [query, *(chunk.search_text for chunk in chunks)]
                )
                query_vector = vectors[0]
                semantic_scores = [
                    _cosine_similarity(query_vector, vector)
                    for vector in vectors[1:]
                ]
            except Exception as exc:
                logger.warning(
                    "V2 memory embedding retrieval unavailable; using keyword fallback: %s",
                    exc,
                )

        ranked: list[MemorySearchHit] = []
        for index, chunk in enumerate(chunks):
            keyword_score = keyword_scores[index]
            semantic_score = (
                semantic_scores[index] if semantic_scores is not None else 0.0
            )
            if semantic_scores is None:
                score = keyword_score
                relevant = keyword_score > 0.0
            else:
                score = (
                    self._semantic_weight * max(semantic_score, 0.0)
                    + self._keyword_weight * keyword_score
                )
                relevant = (
                    score >= self._score_threshold
                    or keyword_score >= 0.18
                )
            if not relevant:
                continue
            score *= 0.9 + 0.1 * _SOURCE_PRIORITY.get(chunk.source, 0.8)
            ranked.append(
                MemorySearchHit(
                    source=chunk.source,
                    text=chunk.text,
                    score=round(score, 6),
                    semantic_score=round(semantic_score, 6),
                    keyword_score=round(keyword_score, 6),
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.score,
                -item.keyword_score,
                -_SOURCE_PRIORITY.get(item.source, 0.0),
                item.text,
            )
        )
        return ranked[: self._top_k]

    def _collect_chunks(self) -> list[MemoryChunk]:
        sources = (
            ("long_term", self._store.read_long_term()),
            ("pending", self._store.read_pending()),
            ("recent_context", self._store.read_recent_context()),
        )
        chunks: list[MemoryChunk] = []
        seen: set[str] = set()
        for source, content in sources:
            for text, search_text in _split_markdown(content):
                normalized = _normalize_text(text)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                chunks.append(
                    MemoryChunk(
                        source=source,
                        text=text,
                        search_text=search_text,
                        ordinal=len(chunks),
                    )
                )
        return chunks

    def _initialize_cache(self) -> None:
        with sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model, text_hash)
                )
                """
            )
            conn.commit()

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        model_key = self._cache_namespace
        hashes = [_text_hash(text) for text in texts]
        cached: dict[str, list[float]] = {}
        with sqlite3.connect(self._cache_path) as conn:
            for text_hash in dict.fromkeys(hashes):
                row = conn.execute(
                    "SELECT vector_json FROM embedding_cache WHERE model=? AND text_hash=?",
                    (model_key, text_hash),
                ).fetchone()
                if row is not None:
                    cached[text_hash] = [float(value) for value in json.loads(row[0])]

        missing_texts: list[str] = []
        missing_hashes: list[str] = []
        for text, text_hash in zip(texts, hashes, strict=True):
            if text_hash not in cached and text_hash not in missing_hashes:
                missing_texts.append(text)
                missing_hashes.append(text_hash)

        if missing_texts:
            vectors = await self._embed(missing_texts)
            if len(vectors) != len(missing_texts):
                raise ValueError("embedding result count does not match input count")
            with sqlite3.connect(self._cache_path) as conn:
                for text_hash, vector in zip(missing_hashes, vectors, strict=True):
                    normalized = [float(value) for value in vector]
                    cached[text_hash] = normalized
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO embedding_cache
                        (model, text_hash, vector_json, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (model_key, text_hash, json.dumps(normalized)),
                    )
                conn.commit()
        return [cached[text_hash] for text_hash in hashes]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is not None:
            return await self._embedder(texts)
        if self._client is None or not self._model:
            raise RuntimeError("embedding provider is not configured")
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in ordered]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


def _split_markdown(content: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    heading = ""
    paragraph: list[str] = []

    def flush() -> None:
        if not paragraph:
            return
        text = " ".join(paragraph).strip()
        paragraph.clear()
        if text:
            search_text = f"{heading} {text}".strip()
            chunks.append((text[:1200], search_text[:1400]))

    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("<!-- consolidation:"):
            flush()
            continue
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
            continue
        if _MARKDOWN_PREFIX_RE.match(line):
            flush()
            text = _MARKDOWN_PREFIX_RE.sub("", line).strip()
            if text:
                chunks.append((text[:1200], f"{heading} {text}".strip()[:1400]))
            continue
        paragraph.append(line)
        if sum(len(part) for part in paragraph) >= 600:
            flush()
    flush()
    return chunks


def _tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    tokens = {match.group(0) for match in _LATIN_TOKEN_RE.finditer(lowered)}
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _keyword_similarity(query: str, candidate: str) -> float:
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0
    score = overlap / math.sqrt(len(query_tokens) * len(candidate_tokens))
    normalized_query = _normalize_text(query)
    normalized_candidate = _normalize_text(candidate)
    if normalized_query and normalized_query in normalized_candidate:
        score += 0.2
    return min(score, 1.0)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_vector = np.asarray(left, dtype=np.float32)
    right_vector = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "HybridMemoryRetriever",
    "MemoryChunk",
    "MemorySearchHit",
]
