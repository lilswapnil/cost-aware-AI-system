"""Uncertainty estimation for cheap model outputs."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class UncertaintyScore:
    score: float
    reason: str


def estimate_uncertainty(text: str, avg_entropy: Optional[float]) -> UncertaintyScore:
    if avg_entropy is not None:
        normalized = min(avg_entropy / 8.0, 1.0)
        return UncertaintyScore(score=normalized, reason="avg_entropy")

    lowered = text.lower()
    if any(phrase in lowered for phrase in ["i am not sure", "i'm not sure", "not certain"]):
        return UncertaintyScore(score=0.9, reason="uncertain_phrase")

    words = re.findall(r"\w+", lowered)
    if not words:
        return UncertaintyScore(score=1.0, reason="empty_output")
    unique_ratio = len(set(words)) / len(words)
    if len(words) < 10:
        return UncertaintyScore(score=0.7, reason="too_short")
    if unique_ratio < 0.4:
        return UncertaintyScore(score=0.8, reason="repetition")

    return UncertaintyScore(score=0.2, reason="heuristic_low")
