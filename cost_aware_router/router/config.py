"""Configuration models for the cost-aware router."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class RouterConfig:
    pricing_path: Path
    cache_path: Path
    cheap_tokenizer_path: Path
    cheap_checkpoint_path: Path
    cheap_device: str
    cheap_max_new_tokens: int
    cheap_temperature: float
    openai_model: str
    openai_timeout_s: int
    default_max_cost_usd: float
    default_max_latency_ms: int
    default_min_quality: float

    @classmethod
    def from_yaml(cls, path: Path) -> "RouterConfig":
        raw: Dict[str, Any] = yaml.safe_load(path.read_text())
        return cls(
            pricing_path=Path(raw["pricing"]["path"]),
            cache_path=Path(raw["cache"]["path"]),
            cheap_tokenizer_path=Path(raw["cheap_model"]["tokenizer_path"]),
            cheap_checkpoint_path=Path(raw["cheap_model"]["checkpoint_path"]),
            cheap_device=raw["cheap_model"].get("device", "cpu"),
            cheap_max_new_tokens=int(raw["cheap_model"].get("max_new_tokens", 128)),
            cheap_temperature=float(raw["cheap_model"].get("temperature", 0.0)),
            openai_model=raw["openai"].get("model", "gpt-4o-mini"),
            openai_timeout_s=int(raw["openai"].get("timeout_s", 30)),
            default_max_cost_usd=float(raw["router"].get("max_cost_usd", 0.02)),
            default_max_latency_ms=int(raw["router"].get("max_latency_ms", 2000)),
            default_min_quality=float(raw["router"].get("min_quality", 0.6)),
        )
