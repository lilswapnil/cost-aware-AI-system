"""Cost estimation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass(frozen=True)
class Pricing:
    input_per_1k: float
    output_per_1k: float


class CostEstimator:
    def __init__(self, pricing_path: Path) -> None:
        raw: Dict[str, Dict[str, float]] = yaml.safe_load(pricing_path.read_text())
        self._pricing = {
            name: Pricing(**values) for name, values in raw.items()
        }

    def estimate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        pricing = self._pricing[model]
        return (tokens_in / 1000) * pricing.input_per_1k + (tokens_out / 1000) * pricing.output_per_1k

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(int(len(text.split()) * 1.3), 1)
