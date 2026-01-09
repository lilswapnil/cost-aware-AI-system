"""Shared types for router outputs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    estimated_cost_usd: float
    quality_proxy: float
    chosen_model: str
    route_reason: str
    cache_hit: bool
    saved_cost_usd: float
