try:
    from .cost_aware_router import AdaptiveLLMRouter, ModelSpec, CostTracker, estimate_tokens
except ImportError:
    from cost_aware_router import AdaptiveLLMRouter, ModelSpec, CostTracker, estimate_tokens

# Define multiple model specs for richer routing
CHEAP_TIER = ModelSpec(
    name="from_scratch_125m",
    cost_per_1k_tokens=0.01,
    avg_latency_ms=100,
    max_context_tokens=1024,
    quality_score=0.5,
)
QUALITY_TIER = ModelSpec(
    name="teacher_gpt2_large",
    cost_per_1k_tokens=0.1,
    avg_latency_ms=400,
    max_context_tokens=1024,
    quality_score=0.7,
)
GPT_3_5 = ModelSpec(
    name="gpt-3.5-turbo",
    cost_per_1k_tokens=2.0,
    avg_latency_ms=600,
    max_context_tokens=4096,
    quality_score=0.8,
)
GPT_4 = ModelSpec(
    name="gpt-4",
    cost_per_1k_tokens=30.0,
    avg_latency_ms=1200,
    max_context_tokens=8192,
    quality_score=0.95,
)
LLAMA_2 = ModelSpec(
    name="llama-2-70b",
    cost_per_1k_tokens=1.0,
    avg_latency_ms=800,
    max_context_tokens=4096,
    quality_score=0.85,
)
MISTRAL = ModelSpec(
    name="mistral-8x7b",
    cost_per_1k_tokens=2.0,
    avg_latency_ms=500,
    max_context_tokens=32000,
    quality_score=0.9,
)

class Router:
    def __init__(self):
        self.models = [CHEAP_TIER, QUALITY_TIER, GPT_3_5, GPT_4, LLAMA_2, MISTRAL]
        self.router = AdaptiveLLMRouter(self.models)
        self.tracker = CostTracker(monthly_budget=100.0)

    def route(self, prompt, max_cost=1.0, max_latency=2.0):
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = 120
        record = self.router.route(prompt, prompt_tokens, completion_tokens, self.tracker)
        within_budget = record.cost <= max_cost
        within_latency = record.latency_ms <= max_latency * 1000
        return {
            "model": record.model_name,
            "cost": record.cost,
            "latency_ms": record.latency_ms,
            "within_budget": within_budget,
            "within_latency": within_latency
        }
