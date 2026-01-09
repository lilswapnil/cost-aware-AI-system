"""Pydantic schemas for the router API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_cost_usd: Optional[float] = Field(default=None, ge=0)
    max_latency_ms: Optional[int] = Field(default=None, ge=0)
    min_quality: Optional[float] = Field(default=None, ge=0, le=1)
    force_model: Optional[str] = Field(default=None, pattern="^(cheap|openai)$")


class GenerateResponse(BaseModel):
    text: str
    chosen_model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    estimated_cost_usd: float
    quality_proxy: float
    route_reason: str
    cache_hit: bool
    saved_cost_usd: float
