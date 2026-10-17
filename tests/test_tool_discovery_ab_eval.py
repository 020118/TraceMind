from __future__ import annotations

import asyncio

from eval.tool_discovery_ab import load_dataset, run_experiment


def test_tool_discovery_ab_deterministic_harness() -> None:
    dataset = load_dataset()
    assert len(dataset["cases"]) == 7

    payload = asyncio.run(run_experiment(mode="deterministic", runs=1))
    baseline = payload["aggregates"]["baseline"]
    deferred = payload["aggregates"]["deferred"]

    assert baseline["attempted_runs"] == 7
    assert baseline["valid_runs"] == 7
    assert baseline["infrastructure_error_rate"] == 0.0
    assert baseline["task_success_rate"] == 1.0
    assert deferred["task_success_rate"] == 1.0
    assert baseline["dynamic_discovery_rate"] == 0.0
    assert deferred["dynamic_discovery_rate"] == 1.0
    assert deferred["schema_load_rate"] == 1.0
    assert deferred["avg_initial_schema_count"] == 1.0
    assert baseline["avg_initial_schema_count"] > deferred["avg_initial_schema_count"]

