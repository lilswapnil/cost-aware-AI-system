"""Demo for the cost-aware adaptive LLM router."""
from src.cost_aware_router import (
    AdaptiveLLMRouter,
    CostTracker,
    ModelSpec,
    estimate_tokens,
    summarize_costs,
)


def main() -> None:
    models = [
        ModelSpec(
            name="gpt-4-class",
            cost_per_1k_tokens=0.06,
            avg_latency_ms=1200,
            max_context_tokens=128000,
            quality_score=0.95,
        ),
        ModelSpec(
            name="gpt-3.5-class",
            cost_per_1k_tokens=0.01,
            avg_latency_ms=600,
            max_context_tokens=16000,
            quality_score=0.7,
        ),
        ModelSpec(
            name="local-7b",
            cost_per_1k_tokens=0.002,
            avg_latency_ms=350,
            max_context_tokens=8000,
            quality_score=0.55,
        ),
    ]

    tracker = CostTracker(monthly_budget=250.0)
    router = AdaptiveLLMRouter(models=models, target_latency_ms=1500)

    prompts = [
        "Summarize last week's support tickets into three bullet points.",
        "Design a multi-step rollout plan and analyze trade-offs for risk mitigation.",
        "Rewrite this email to sound friendly and professional.",
    ]

    for prompt in prompts:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = 120
        record = router.route(prompt, prompt_tokens, completion_tokens, tracker)
        print(
            f"Prompt: {prompt}\n"
            f"-> Routed to {record.model_name} | cost ${record.cost:.4f} | "
            f"latency {record.latency_ms}ms\n"
        )

    baseline = models[0]
    print(summarize_costs(tracker, baseline))


if __name__ == "__main__":
    main()
