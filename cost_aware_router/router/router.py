"""Main router logic for cost-aware routing."""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from cost_aware_router.adapters.cheap_model import CheapModelAdapter
from cost_aware_router.adapters.openai_model import OpenAIModelAdapter
from cost_aware_router.cache.cache import PromptCache
from cost_aware_router.estimators.cost import CostEstimator
from cost_aware_router.estimators.latency import within_sla
from cost_aware_router.estimators.quality import score_quality
from cost_aware_router.estimators.uncertainty import estimate_uncertainty
from cost_aware_router.router.config import RouterConfig
from cost_aware_router.router.policy import RoutingPolicy
from cost_aware_router.router.types import GenerationResult


logger = logging.getLogger(__name__)


class CostAwareRouter:
    def __init__(
        self,
        cheap_adapter: CheapModelAdapter,
        openai_adapter: OpenAIModelAdapter,
        cache: PromptCache,
        cost_estimator: CostEstimator,
        policy: RoutingPolicy,
        default_max_cost_usd: float,
        default_max_latency_ms: int,
        default_min_quality: float,
    ) -> None:
        self.cheap_adapter = cheap_adapter
        self.openai_adapter = openai_adapter
        self.cache = cache
        self.cost_estimator = cost_estimator
        self.policy = policy
        self.default_max_cost_usd = default_max_cost_usd
        self.default_max_latency_ms = default_max_latency_ms
        self.default_min_quality = default_min_quality

    @classmethod
    def from_config(cls, config: RouterConfig) -> "CostAwareRouter":
        cheap_adapter = CheapModelAdapter(
            tokenizer_path=str(config.cheap_tokenizer_path),
            checkpoint_path=str(config.cheap_checkpoint_path),
            device=config.cheap_device,
            max_new_tokens=config.cheap_max_new_tokens,
            temperature=config.cheap_temperature,
        )
        openai_adapter = OpenAIModelAdapter(
            model=config.openai_model,
            timeout_s=config.openai_timeout_s,
        )
        cache = PromptCache(config.cache_path)
        cost_estimator = CostEstimator(config.pricing_path)
        policy = RoutingPolicy()
        return cls(
            cheap_adapter=cheap_adapter,
            openai_adapter=openai_adapter,
            cache=cache,
            cost_estimator=cost_estimator,
            policy=policy,
            default_max_cost_usd=config.default_max_cost_usd,
            default_max_latency_ms=config.default_max_latency_ms,
            default_min_quality=config.default_min_quality,
        )

    def generate(
        self,
        prompt: str,
        max_cost_usd: Optional[float] = None,
        max_latency_ms: Optional[int] = None,
        min_quality: Optional[float] = None,
        force_model: Optional[str] = None,
    ) -> GenerationResult:
        max_cost_usd = max_cost_usd if max_cost_usd is not None else self.default_max_cost_usd
        max_latency_ms = (
            max_latency_ms if max_latency_ms is not None else self.default_max_latency_ms
        )
        min_quality = min_quality if min_quality is not None else self.default_min_quality

        constraints = {
            "max_cost_usd": max_cost_usd,
            "max_latency_ms": max_latency_ms,
            "min_quality": min_quality,
            "force_model": force_model,
        }

        if force_model is None:
            cached = self.cache.get(prompt, constraints)
            if cached:
                cached_response = cached.response
                cached_response["cache_hit"] = True
                logger.info("Cache hit for prompt")
                return GenerationResult(**cached_response)

        if force_model == "openai":
            return self._run_openai(prompt, "force_model=openai", False, max_cost_usd)
        if force_model == "cheap":
            return self._run_cheap(prompt, "force_model=cheap", False, max_cost_usd, max_latency_ms, min_quality)

        decision = self.policy.decide(prompt)
        if decision.action == "openai":
            estimated_openai_cost = self._estimate_openai_cost(prompt, self.cheap_adapter.max_new_tokens)
            if estimated_openai_cost <= max_cost_usd:
                result = self._run_openai(prompt, decision.reason, False, max_cost_usd)
            else:
                result = self._run_cheap(
                    prompt,
                    f"{decision.reason}; openai_cost_exceeds_limit",
                    False,
                    max_cost_usd,
                    max_latency_ms,
                    min_quality,
                )
        else:
            result = self._run_cheap(
                prompt,
                decision.reason,
                False,
                max_cost_usd,
                max_latency_ms,
                min_quality,
            )

        if result.cache_hit is False:
            self.cache.set(prompt, constraints, asdict(result))
        return result

    def _estimate_openai_cost(self, prompt: str, max_new_tokens: int) -> float:
        tokens_in = self.cost_estimator.estimate_tokens(prompt)
        tokens_out = max_new_tokens
        return self.cost_estimator.estimate_cost("openai", tokens_in, tokens_out)

    def _run_openai(
        self, prompt: str, reason: str, cache_hit: bool, max_cost_usd: float
    ) -> GenerationResult:
        generation = self.openai_adapter.generate(prompt)
        quality = score_quality(prompt, generation.text)
        cost = self.cost_estimator.estimate_cost("openai", generation.tokens_in, generation.tokens_out)
        if cost > max_cost_usd:
            reason = f"{reason}; openai_cost_exceeds_limit"
        return GenerationResult(
            text=generation.text,
            chosen_model="openai",
            tokens_in=generation.tokens_in,
            tokens_out=generation.tokens_out,
            latency_ms=generation.latency_ms,
            estimated_cost_usd=cost,
            quality_proxy=quality.score,
            route_reason=reason,
            cache_hit=cache_hit,
            saved_cost_usd=0.0,
        )

    def _run_cheap(
        self,
        prompt: str,
        reason: str,
        cache_hit: bool,
        max_cost_usd: float,
        max_latency_ms: int,
        min_quality: float,
    ) -> GenerationResult:
        generation = self.cheap_adapter.generate(prompt)
        quality = score_quality(prompt, generation.text)
        uncertainty = estimate_uncertainty(generation.text, generation.avg_entropy)
        cost = self.cost_estimator.estimate_cost("cheap", generation.tokens_in, generation.tokens_out)
        openai_baseline_cost = self.cost_estimator.estimate_cost(
            "openai", generation.tokens_in, generation.tokens_out
        )
        saved_cost = max(openai_baseline_cost - cost, 0.0)

        escalation_reasons = []
        if uncertainty.score >= 0.6:
            escalation_reasons.append(f"uncertainty_high({uncertainty.reason})")
        if quality.score < min_quality:
            escalation_reasons.append("quality_below_min")
        if not within_sla(generation.latency_ms, max_latency_ms):
            escalation_reasons.append("latency_exceeds_sla")

        if escalation_reasons and openai_baseline_cost <= max_cost_usd:
            new_reason = f"{reason}; escalate={'|'.join(escalation_reasons)}"
            return self._run_openai(prompt, new_reason, cache_hit, max_cost_usd)

        route_reason = reason
        if escalation_reasons and openai_baseline_cost > max_cost_usd:
            route_reason = f"{reason}; escalation_blocked_by_cost"

        return GenerationResult(
            text=generation.text,
            chosen_model="cheap",
            tokens_in=generation.tokens_in,
            tokens_out=generation.tokens_out,
            latency_ms=generation.latency_ms,
            estimated_cost_usd=cost,
            quality_proxy=quality.score,
            route_reason=route_reason,
            cache_hit=cache_hit,
            saved_cost_usd=saved_cost,
        )
