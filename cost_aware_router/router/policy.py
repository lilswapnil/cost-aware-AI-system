"""Rule-based routing policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cost_aware_router.router.prompt_analyzer import PromptAnalyzer, PromptComplexity


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str
    complexity: PromptComplexity


class RoutingPolicy:
    def __init__(self, analyzer: Optional[PromptAnalyzer] = None) -> None:
        self._analyzer = analyzer or PromptAnalyzer()

    def decide(self, prompt: str) -> PolicyDecision:
        complexity = self._analyzer.score(prompt)
        if complexity.label == "high":
            return PolicyDecision(
                action="openai",
                reason=f"prompt_complexity_high(score={complexity.score:.2f})",
                complexity=complexity,
            )
        return PolicyDecision(
            action="cheap",
            reason=f"prompt_complexity_low(score={complexity.score:.2f})",
            complexity=complexity,
        )
