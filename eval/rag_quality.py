"""Offline end-to-end RAG quality benchmark.

The benchmark creates a temporary workspace, indexes a small labelled corpus,
and runs the production ``RAGService`` twice: once with plain RRF results and
once with the current MMR/diversity policy. It never calls an external LLM or
Embedding API, so it is suitable for CI and regression checks.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from agent.rag.service import RAGService
from agent.rag.store import SQLiteRAGStore

DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "rag_quality_smoke.json"
DEFAULT_JSON = Path(__file__).resolve().parent / "results" / "rag_quality_smoke.json"
DEFAULT_MARKDOWN = Path(__file__).resolve().parent / "results" / "rag_quality_smoke.md"
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


class HashEmbedding:
    """Deterministic local embedding used only for offline regression tests."""

    model_name = "offline-hash-embedding-64"
    dimensions = 64

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def close(self) -> None:
        return None

    @classmethod
    def _embed_one(cls, text: str) -> list[float]:
        vector = [0.0] * cls.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % cls.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


async def run_benchmark(
    dataset_path: str | Path = DEFAULT_DATASET,
    *,
    top_k: int = 3,
    improved_max_per_document: int = 2,
) -> dict[str, Any]:
    """Run baseline-vs-improved retrieval over a labelled local corpus."""
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    top_k = max(1, min(int(top_k), 20))
    improved_max_per_document = max(0, min(int(improved_max_per_document), 20))

    # Keep the temporary corpus inside the project workspace. Some managed
    # Windows environments deny Python access to user-level Temp folders or
    # reject the restrictive mode used by ``TemporaryDirectory``.
    workspace = Path(__file__).resolve().parents[1] / f".tracemind-rag-eval-{uuid.uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        service = RAGService(
            workspace=workspace,
            store=SQLiteRAGStore(workspace / "knowledge" / "rag.db"),
            embedder=HashEmbedding(),
            top_k=top_k,
            diversify_results=True,
            max_results_per_document=improved_max_per_document,
            search_cache_size=0,
        )
        await service.initialize()
        try:
            for document in dataset.get("documents", []):
                path = workspace / str(document["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(document.get("content", "")), encoding="utf-8")
                await service.ingest_file(str(document["path"]), title=str(document.get("title", "")))

            baseline = await _evaluate_variant(
                service,
                dataset.get("queries", []),
                top_k=top_k,
                diversify=False,
                max_per_document=0,
            )
            improved = await _evaluate_variant(
                service,
                dataset.get("queries", []),
                top_k=top_k,
                diversify=True,
                max_per_document=improved_max_per_document,
            )
        finally:
            await service.close()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return {
        "dataset": {
            "name": dataset.get("name", Path(dataset_path).stem),
            "path": str(dataset_path),
            "documents": len(dataset.get("documents", [])),
            "queries": len(dataset.get("queries", [])),
            "top_k": top_k,
        },
        "embedding": HashEmbedding.model_name,
        "baseline": baseline,
        "improved": improved,
        "delta": _metric_delta(baseline["metrics"], improved["metrics"]),
    }


async def _evaluate_variant(
    service: RAGService,
    queries: list[dict[str, Any]],
    *,
    top_k: int,
    diversify: bool,
    max_per_document: int,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for case in queries:
        started = time.perf_counter()
        hits = await service.search(
            str(case.get("query", "")),
            top_k=top_k,
            diversify=diversify,
            max_per_document=max_per_document,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        relevance = {
            str(path): float(score)
            for path, score in dict(case.get("relevance", {})).items()
        }
        source_paths = _unique_in_order(hit.document.source_path for hit in hits)
        details.append(
            {
                "id": str(case.get("id", "")),
                "query": str(case.get("query", "")),
                "retrieved_sources": source_paths,
                "retrieved_chunk_ids": [hit.chunk.id for hit in hits],
                "retrieved_scores": [hit.score for hit in hits],
                "retrieved_keyword_scores": [hit.keyword_score for hit in hits],
                "retrieved_semantic_scores": [hit.semantic_score for hit in hits],
                "recall_at_k": _recall(source_paths, relevance),
                "mrr": _mrr(source_paths, relevance),
                "ndcg_at_k": _ndcg(source_paths, relevance, top_k),
                "citation_coverage": _citation_coverage(hits),
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )

    metrics = {
        "query_count": len(details),
        "recall_at_k": _mean(detail["recall_at_k"] for detail in details),
        "mrr": _mean(detail["mrr"] for detail in details),
        "ndcg_at_k": _mean(detail["ndcg_at_k"] for detail in details),
        "citation_coverage": _mean(
            detail["citation_coverage"] for detail in details
        ),
        "empty_result_rate": _mean(
            1.0 if not detail["retrieved_sources"] else 0.0 for detail in details
        ),
        "distinct_sources_at_k": _mean(
            float(len(detail["retrieved_sources"])) for detail in details
        ),
        "average_latency_ms": _mean(detail["elapsed_ms"] for detail in details),
    }
    return {"config": {"diversify": diversify, "max_per_document": max_per_document}, "metrics": metrics, "queries": details}


def render_markdown(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    baseline = payload["baseline"]
    improved = payload["improved"]
    delta = payload["delta"]
    lines = [
        f"# {dataset['name']}",
        "",
        "离线、确定性的 RAG 检索质量对比；不调用外部 LLM 或 Embedding API。",
        "",
        f"- 文档数：{dataset['documents']}",
        f"- 查询数：{dataset['queries']}",
        f"- Top-K：{dataset['top_k']}",
        f"- Embedding：`{payload['embedding']}`",
        "",
        "| 指标 | Baseline（RRF） | Improved（RRF + MMR） | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "citation_coverage",
        "empty_result_rate",
        "distinct_sources_at_k",
        "average_latency_ms",
    ):
        lines.append(
            f"| {name} | {baseline['metrics'][name]} | "
            f"{improved['metrics'][name]} | {delta[name]} |"
        )
    lines.extend(["", "## Query Details", ""])
    lines.append("| Query | Baseline sources | Improved sources | Improved nDCG@K |")
    lines.append("|---|---|---|---:|")
    for before, after in zip(baseline["queries"], improved["queries"], strict=True):
        lines.append(
            f"| {before['id']} | {', '.join(before['retrieved_sources']) or '无'} | "
            f"{', '.join(after['retrieved_sources']) or '无'} | {after['ndcg_at_k']} |"
        )
    return "\n".join(lines) + "\n"


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.findall(text.lower()):
        if all("\u4e00" <= char <= "\u9fff" for char in match):
            tokens.update(match)
            tokens.update(match[index : index + 2] for index in range(len(match) - 1))
        else:
            tokens.add(match)
    return tokens


def _unique_in_order(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _recall(retrieved: list[str], relevance: dict[str, float]) -> float:
    if not relevance:
        return 0.0
    return round(len(set(retrieved) & set(relevance)) / len(relevance), 6)


def _mrr(retrieved: list[str], relevance: dict[str, float]) -> float:
    for index, source in enumerate(retrieved, start=1):
        if relevance.get(source, 0.0) > 0:
            return round(1.0 / index, 6)
    return 0.0


def _ndcg(retrieved: list[str], relevance: dict[str, float], top_k: int) -> float:
    def gain(score: float) -> float:
        return (2.0**score) - 1.0

    actual = sum(
        gain(relevance.get(source, 0.0)) / math.log2(index + 2)
        for index, source in enumerate(retrieved[:top_k])
    )
    ideal_scores = sorted((score for score in relevance.values() if score > 0), reverse=True)
    ideal = sum(gain(score) / math.log2(index + 2) for index, score in enumerate(ideal_scores[:top_k]))
    return round(actual / ideal, 6) if ideal else 0.0


def _citation_coverage(hits: list[Any]) -> float:
    if not hits:
        return 0.0
    valid = sum(
        1
        for hit in hits
        if hit.citation and hit.document.source_path
    )
    return round(valid / len(hits), 6)


def _mean(values: Any) -> float:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else 0.0


def _metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        name: round(float(after.get(name, 0.0)) - float(before.get(name, 0.0)), 6)
        for name in before
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-per-document", type=int, default=2)
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()
    payload = asyncio.run(
        run_benchmark(
            args.dataset,
            top_k=args.top_k,
            improved_max_per_document=args.max_per_document,
        )
    )
    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()


__all__ = ["run_benchmark", "render_markdown"]
