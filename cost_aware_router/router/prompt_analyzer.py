"""Prompt complexity analyzer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptComplexity:
    score: float
    label: str


class PromptAnalyzer:
    COMPLEXITY_HINTS = (
        "analyze",
        "strategy",
        "multi-step",
        "reason",
        "design",
        "architecture",
        "trade-off",
        "evaluation",
        "proof",
        "optimize",
    )

    def score(self, prompt: str) -> PromptComplexity:
        tokens = prompt.lower().split()
        length_score = min(len(tokens) / 200, 1.0)
        hint_score = sum(1 for hint in self.COMPLEXITY_HINTS if hint in prompt.lower()) / len(
            self.COMPLEXITY_HINTS
        )
        score = min(length_score * 0.6 + hint_score * 0.4, 1.0)
        label = "high" if score >= 0.6 else "low"
        return PromptComplexity(score=score, label=label)
