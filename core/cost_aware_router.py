"""Cost-aware routing primitives for selecting LLMs based on quality, cost, and latency."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cost_per_1k_tokens: float
    avg_latency_ms: int
    max_context_tokens: int
    quality_score: float


@dataclass
class CallRecord:
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency_ms: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CostTracker:
    """Tracks per-call spend, aggregates totals, and estimates savings."""

    def __init__(self, monthly_budget: float, currency: str = "USD") -> None:
        self.monthly_budget = monthly_budget
        self.currency = currency
        self.records: List[CallRecord] = []

    def record_call(
        self,
        model: ModelSpec,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
    ) -> CallRecord:
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000) * model.cost_per_1k_tokens
        record = CallRecord(
            model_name=model.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency_ms=latency_ms,
        )
        self.records.append(record)
        return record

    def total_cost(self) -> float:
        return sum(record.cost for record in self.records)

    def remaining_budget(self) -> float:
        return max(self.monthly_budget - self.total_cost(), 0.0)

    def savings_vs_baseline(self, baseline_model: ModelSpec) -> float:
        baseline_cost = sum(
            (record.total_tokens / 1000) * baseline_model.cost_per_1k_tokens
            for record in self.records
        )
        return max(baseline_cost - self.total_cost(), 0.0)


class PromptScorer:
    """Lightweight prompt scoring to estimate complexity."""

    COMPLEXITY_HINTS = (
        "analyze",
        "strategy",
        "multi-step",
        "reason",
        "design",
        "architecture",
        "trade-off",
        "evaluation",
    )

    def score(self, prompt: str) -> float:
        tokens = prompt.lower().split()
        length_score = min(len(tokens) / 200, 1.0)
        hint_score = sum(1 for hint in self.COMPLEXITY_HINTS if hint in prompt.lower()) / len(
            self.COMPLEXITY_HINTS
        )
        return min(length_score * 0.6 + hint_score * 0.4, 1.0)


class AdaptiveLLMRouter:
    """Routes prompts to the most cost-efficient model that meets quality needs."""

    def __init__(
        self,
        models: Iterable[ModelSpec],
        scorer: Optional[PromptScorer] = None,
        target_latency_ms: Optional[int] = None,
    ) -> None:
        self.models = sorted(models, key=lambda model: model.quality_score)
        self.scorer = scorer or PromptScorer()
        self.target_latency_ms = target_latency_ms

    def select_model(
        self,
        prompt: str,
        min_quality: Optional[float] = None,
        max_cost_per_1k: Optional[float] = None,
        max_context_tokens: Optional[int] = None,
    ) -> ModelSpec:
        required_quality = min_quality if min_quality is not None else self.scorer.score(prompt)
        candidates = [
            model
            for model in self.models
            if model.quality_score >= required_quality
            and (max_cost_per_1k is None or model.cost_per_1k_tokens <= max_cost_per_1k)
            and (max_context_tokens is None or model.max_context_tokens >= max_context_tokens)
        ]

        if self.target_latency_ms is not None:
            candidates = [
                model for model in candidates if model.avg_latency_ms <= self.target_latency_ms
            ] or candidates

        if not candidates:
            return max(self.models, key=lambda model: model.quality_score)

        return min(candidates, key=lambda model: (model.cost_per_1k_tokens, model.avg_latency_ms))

    def route(
        self,
        prompt: str,
        prompt_tokens: int,
        completion_tokens: int,
        tracker: CostTracker,
    ) -> CallRecord:
        budget_pressure = tracker.remaining_budget() / tracker.monthly_budget
        max_cost = None
        if budget_pressure < 0.2:
            max_cost = min(model.cost_per_1k_tokens for model in self.models)

        model = self.select_model(
            prompt,
            max_cost_per_1k=max_cost,
            max_context_tokens=prompt_tokens + completion_tokens,
        )
        record = tracker.record_call(
            model,
            prompt_tokens,
            completion_tokens,
            latency_ms=model.avg_latency_ms,
        )
        return record


def estimate_tokens(text: str) -> int:
    return max(int(len(text.split()) * 1.3), 1)


def summarize_costs(tracker: CostTracker, baseline: ModelSpec) -> str:
    return (
        f"Total spend: {tracker.total_cost():.2f} {tracker.currency} | "
        f"Savings vs {baseline.name}: {tracker.savings_vs_baseline(baseline):.2f} {tracker.currency}"
    )
