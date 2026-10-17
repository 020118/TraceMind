"""Measure Qdrant ANN retrieval against the former in-process exact scan.

The generated vectors are deterministic and the temporary Qdrant collection is
deleted by default. This benchmark does not call an LLM or an Embedding API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from statistics import fmean

import numpy as np
from qdrant_client import AsyncQdrantClient, models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://qdrant:6333"))
    parser.add_argument("--vectors", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.vectors, args.dimensions)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query_source = rng.choice(args.vectors, size=args.queries, replace=False)
    queries = vectors[query_source] + rng.normal(
        scale=0.01, size=(args.queries, args.dimensions)
    ).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    tuple_vectors = [tuple(float(value) for value in row) for row in vectors]

    collection = f"agent_rag_benchmark_{int(time.time())}"
    client = AsyncQdrantClient(url=args.url, timeout=30)
    ingestion_started = time.perf_counter()
    try:
        await client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=args.dimensions,
                distance=models.Distance.COSINE,
            ),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
        )
        for start in range(0, args.vectors, 256):
            await client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(id=index, vector=vectors[index].tolist())
                    for index in range(start, min(start + 256, args.vectors))
                ],
                wait=True,
            )
        ingestion_ms = (time.perf_counter() - ingestion_started) * 1000
        await _wait_for_index(client, collection)

        # Warm the HTTP connection and Qdrant search path outside the timing sample.
        await client.query_points(
            collection_name=collection,
            query=queries[0].tolist(),
            limit=args.top_k,
        )

        qdrant_ms: list[float] = []
        current_exact_ms: list[float] = []
        vectorized_exact_ms: list[float] = []
        recalls: list[float] = []
        for query in queries:
            started = time.perf_counter()
            exact_scores = vectors @ query
            exact_ids = np.argpartition(exact_scores, -args.top_k)[-args.top_k:]
            exact_ids = exact_ids[np.argsort(exact_scores[exact_ids])[::-1]].tolist()
            vectorized_exact_ms.append((time.perf_counter() - started) * 1000)

            # This mirrors the previous application path: convert and compare every
            # stored tuple separately instead of using a vector index.
            started = time.perf_counter()
            loop_scores = []
            query_norm = float(np.linalg.norm(query))
            for index, stored in enumerate(tuple_vectors):
                vector = np.asarray(stored, dtype=np.float32)
                denominator = float(np.linalg.norm(vector) * query_norm)
                score = float(np.dot(query, vector) / denominator) if denominator else 0.0
                loop_scores.append((score, index))
            loop_scores.sort(reverse=True)
            current_exact_ms.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            response = await client.query_points(
                collection_name=collection,
                query=query.tolist(),
                limit=args.top_k,
                with_payload=False,
                with_vectors=False,
            )
            qdrant_ms.append((time.perf_counter() - started) * 1000)
            qdrant_ids = {int(point.id) for point in response.points}
            recalls.append(len(set(exact_ids) & qdrant_ids) / len(exact_ids))

        qdrant_summary = _summary(qdrant_ms)
        current_summary = _summary(current_exact_ms)
        vectorized_summary = _summary(vectorized_exact_ms)
        return {
            "dataset": {
                "vectors": args.vectors,
                "dimensions": args.dimensions,
                "queries": args.queries,
                "top_k": args.top_k,
                "seed": args.seed,
            },
            "ingestion_ms": round(ingestion_ms, 3),
            "recall_at_k": round(fmean(recalls), 4),
            "qdrant_ms": qdrant_summary,
            "current_application_exact_scan_ms": current_summary,
            "vectorized_numpy_ground_truth_ms": vectorized_summary,
            "speedup_vs_current_scan": round(
                current_summary["average"] / qdrant_summary["average"], 3
            ),
            "note": (
                "Qdrant和SQLite基准都属于Dense检索；这里比较的是Qdrant向量索引"
                "与迁移前应用层逐条精确扫描。"
            ),
        }
    finally:
        if not args.keep:
            try:
                await client.delete_collection(collection)
            except Exception:
                pass
        await client.close()


async def _wait_for_index(client: AsyncQdrantClient, collection: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        info = await client.get_collection(collection)
        if int(info.indexed_vectors_count or 0) >= int(info.points_count or 0):
            return
        await asyncio.sleep(0.5)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "average": round(fmean(values), 3),
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2))
