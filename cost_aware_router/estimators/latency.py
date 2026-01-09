"""Latency estimation helpers."""
from __future__ import annotations


def within_sla(latency_ms: int, max_latency_ms: int) -> bool:
    return latency_ms <= max_latency_ms
