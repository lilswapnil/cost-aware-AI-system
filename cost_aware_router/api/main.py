"""FastAPI app for the cost-aware router."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from cost_aware_router.api.schemas import GenerateRequest, GenerateResponse
from cost_aware_router.router.config import RouterConfig
from cost_aware_router.router.router import CostAwareRouter


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Cost-Aware AI Router", version="1.0")
router_instance: CostAwareRouter | None = None


@app.on_event("startup")
def startup() -> None:
    global router_instance
    config = RouterConfig.from_yaml(Path("cost_aware_router/configs/default.yaml"))
    router_instance = CostAwareRouter.from_config(config)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    assert router_instance is not None
    result = router_instance.generate(
        prompt=request.prompt,
        max_cost_usd=request.max_cost_usd,
        max_latency_ms=request.max_latency_ms,
        min_quality=request.min_quality,
        force_model=request.force_model,
    )
    return GenerateResponse(**result.model_dump())
