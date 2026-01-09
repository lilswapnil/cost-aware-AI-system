from dataclasses import dataclass

@dataclass
class Request:
    prompt: str
    max_cost: float = 1.0
    max_latency: float = 2.0

@dataclass
class Response:
    model: str
    cost: float
    latency_ms: int
    within_budget: bool
    within_latency: bool
