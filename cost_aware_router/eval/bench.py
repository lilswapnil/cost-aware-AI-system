"""Evaluation harness for cost-aware routing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from cost_aware_router.router.config import RouterConfig
from cost_aware_router.router.router import CostAwareRouter


@dataclass
class EvalResult:
    total_cost_usd: float
    sla_hit_rate: float
    avg_quality: float


def load_prompts(path: Path) -> List[Dict[str, object]]:
    prompts: List[Dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        prompts.append(json.loads(line))
    return prompts


def run_mode(
    router: CostAwareRouter,
    prompts: List[Dict[str, object]],
    force_model: Optional[str],
    default_latency: int,
) -> EvalResult:
    results = []
    sla_hits = 0
    for item in prompts:
        prompt = str(item["prompt"])
        max_latency_ms = int(item.get("max_latency_ms", default_latency))
        result = router.generate(
            prompt=prompt,
            max_cost_usd=item.get("max_cost_usd"),
            max_latency_ms=max_latency_ms,
            min_quality=item.get("min_quality"),
            force_model=force_model,
        )
        results.append(result)
        if result.latency_ms <= max_latency_ms:
            sla_hits += 1
    total_cost = sum(r.estimated_cost_usd for r in results)
    avg_quality = sum(r.quality_proxy for r in results) / max(len(results), 1)
    sla_hit_rate = sla_hits / max(len(results), 1)
    return EvalResult(total_cost_usd=total_cost, sla_hit_rate=sla_hit_rate, avg_quality=avg_quality)


def main() -> None:
    config = RouterConfig.from_yaml(Path("cost_aware_router/configs/default.yaml"))
    router = CostAwareRouter.from_config(config)
    prompts = load_prompts(Path("cost_aware_router/eval/prompts.jsonl"))

    cheap_results = run_mode(router, prompts, force_model="cheap", default_latency=config.default_max_latency_ms)
    openai_results = run_mode(router, prompts, force_model="openai", default_latency=config.default_max_latency_ms)
    routed_results = run_mode(router, prompts, force_model=None, default_latency=config.default_max_latency_ms)

    cost_reduction_pct = 0.0
    if openai_results.total_cost_usd > 0:
        cost_reduction_pct = (
            (openai_results.total_cost_usd - routed_results.total_cost_usd)
            / openai_results.total_cost_usd
        ) * 100
    quality_retention = 1.0
    if openai_results.avg_quality > 0:
        quality_retention = routed_results.avg_quality / openai_results.avg_quality

    summary = {
        "cheap_only": cheap_results.__dict__,
        "openai_only": openai_results.__dict__,
        "routed": {
            **routed_results.__dict__,
            "cost_reduction_pct": cost_reduction_pct,
            "quality_retention": quality_retention,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
