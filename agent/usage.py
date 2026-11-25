"""Shared model usage recording for auxiliary providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.reliability import ReliabilityStore


class ModelUsageRecorder:
    """Create lightweight callbacks for clients that run outside AgentLoop."""

    def __init__(self, store: ReliabilityStore) -> None:
        self._store = store

    def callback(
        self,
        *,
        model: str,
        purpose: str,
        input_price_per_million_usd: float = 0.0,
        output_price_per_million_usd: float = 0.0,
    ) -> Callable[[dict[str, Any]], None]:
        def record(event: dict[str, Any]) -> None:
            prompt_tokens = int(event.get("prompt_tokens", 0) or 0)
            completion_tokens = int(event.get("completion_tokens", 0) or 0)
            total_tokens = int(
                event.get("total_tokens", 0) or prompt_tokens + completion_tokens
            )
            cost = (
                prompt_tokens * max(0.0, float(input_price_per_million_usd))
                + completion_tokens * max(0.0, float(output_price_per_million_usd))
            ) / 1_000_000
            try:
                self._store.record_model_usage(
                    trace_id=str(event.get("trace_id", "")),
                    session_key=str(event.get("session_key", "")),
                    model=model,
                    purpose=purpose,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=cost,
                )
            except Exception:
                # Usage tracking is observability; it must never break a turn.
                return

        return record
