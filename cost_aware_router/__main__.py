"""CLI entrypoint for the cost-aware router."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cost_aware_router.router.router import CostAwareRouter
from cost_aware_router.router.config import RouterConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cost-aware LLM router")
    parser.add_argument("--config", default="cost_aware_router/configs/default.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--max-latency-ms", type=int, default=None)
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--force-model", choices=["cheap", "openai"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RouterConfig.from_yaml(Path(args.config))
    router = CostAwareRouter.from_config(config)
    result = router.generate(
        prompt=args.prompt,
        max_cost_usd=args.max_cost_usd,
        max_latency_ms=args.max_latency_ms,
        min_quality=args.min_quality,
        force_model=args.force_model,
    )
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
