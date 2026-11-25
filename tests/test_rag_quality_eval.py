from __future__ import annotations

import asyncio

from eval.rag_quality import DEFAULT_DATASET, run_benchmark


def test_rag_quality_benchmark_is_deterministic_and_reports_delta() -> None:
    first = asyncio.run(run_benchmark(DEFAULT_DATASET, top_k=3))
    second = asyncio.run(run_benchmark(DEFAULT_DATASET, top_k=3))

    assert first["dataset"]["documents"] == 3
    assert first["dataset"]["queries"] == 4
    assert first["baseline"]["metrics"]["query_count"] == 4
    assert first["improved"]["metrics"]["citation_coverage"] == 1.0
    assert first["improved"]["metrics"]["distinct_sources_at_k"] >= (
        first["baseline"]["metrics"]["distinct_sources_at_k"]
    )
    assert first["delta"].keys() == first["baseline"]["metrics"].keys()
    assert [item["retrieved_sources"] for item in first["baseline"]["queries"]] == [
        item["retrieved_sources"] for item in second["baseline"]["queries"]
    ]
