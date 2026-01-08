# Cost-Aware AI System — Adaptive LLM Router

This repo is a **cost-aware AI system** that routes prompts to the most cost-efficient model while balancing **quality**, **latency**, and **budget**. It emphasizes business outcomes like monthly spend tracking and savings vs. a baseline model.

## What it is

**Adaptive LLM Router**
- Uses high-quality (expensive) models only when needed.
- Falls back to cheaper models for low-complexity requests.
- Tracks monthly spend and estimated savings vs. a baseline.

## Why it matters
Hiring managers care about **budget impact**. This system is designed to show how you:
- Optimize LLM calls for cost and latency.
- Build decision logic that maps directly to real GenAI systems.
- Track savings and budget usage over time.

## Core components

- **Model Router:** Chooses the best model given cost, latency, and quality constraints.
- **Cost Metrics:** Records per-call usage and monthly spend.
- **Prompt Scoring:** Estimates complexity to determine the minimum viable model.

## Quick start

```bash
python scripts/adaptive_router_demo.py
```

### Example output
```
Prompt: Summarize last week's support tickets into three bullet points.
-> Routed to local-7b | cost $0.0005 | latency 350ms

Prompt: Design a multi-step rollout plan and analyze trade-offs for risk mitigation.
-> Routed to gpt-4-class | cost $0.0108 | latency 1200ms

Total spend: 0.01 USD | Savings vs gpt-4-class: 0.01 USD
```

## Implementation notes

- **Routing logic** lives in `src/cost_aware_router.py`.
- **Demo flow** lives in `scripts/adaptive_router_demo.py`.

### Extend it

You can expand this into a production-grade router by adding:
- Model availability checks and retries.
- Real token counting using a tokenizer.
- Latency percentiles and throughput metrics.
- A live dashboard for spend tracking.

---

If you want to plug in real APIs, swap the demo models in `scripts/adaptive_router_demo.py` with your provider-specific specs and cost rates.
