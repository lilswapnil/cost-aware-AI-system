
"""Demo for the cost-aware adaptive LLM router."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cost_aware_router import (
    AdaptiveLLMRouter,
    CostTracker,
    ModelSpec,
    estimate_tokens,
    summarize_costs,
)
from openai_api import call_openai_chat


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
        ModelSpec(
            name="mistral-8x7b",
            cost_per_1k_tokens=0.004,
            avg_latency_ms=500,
            max_context_tokens=32000,
            quality_score=0.65,
        ),
        ModelSpec(
            name="llama-2-70b",
            cost_per_1k_tokens=0.008,
            avg_latency_ms=800,
            max_context_tokens=4096,
            quality_score=0.75,
        ),
        ModelSpec(
            name="falcon-40b",
            cost_per_1k_tokens=0.006,
            avg_latency_ms=700,
            max_context_tokens=8192,
            quality_score=0.72,
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
        print(f"Prompt: {prompt}")
        # If routed to an OpenAI model, call the API
        if record.model_name == "gpt-4-class":
            api_model = "gpt-4"
        elif record.model_name == "gpt-3.5-class":
            api_model = "gpt-3.5-turbo"
        else:
            api_model = None

        if api_model:
            try:
                response = call_openai_chat(api_model, prompt, max_tokens=completion_tokens)
                output = response["choices"][0]["message"]["content"]
                print(f"-> [OpenAI] {api_model} output: {output}")
            except Exception as e:
                print(f"-> [OpenAI] API call failed: {e}")

        print(
            f"-> Routed to {record.model_name} | cost ${record.cost:.4f} | "
            f"latency {record.latency_ms}ms\n"
        )

    baseline = models[0]
    print(summarize_costs(tracker, baseline))


if __name__ == "__main__":
    main()
