# Cost-Aware LLM Routing System

## What it does

**Cost-aware routing for LLMs:**
- Routes each prompt to the optimal model (cheap or high-quality) based on complexity, cost, and SLA constraints.
- Tracks and reports cost, latency, and quality retention for every request.
- Delivers measurable savings and SLA compliance for real-world GenAI use cases.

## Architecture

```
┌────────────┐      ┌──────────────┐      ┌─────────────┐
│  User/API  ├─────►  Router Core  ├─────►  Model Pool  │
└────────────┘      └──────────────┘      └─────────────┘
				│                  │                    │
				▼                  ▼                    ▼
	 Prompt scoring   Cost/quality/latency   CHEAP_TIER, QUALITY_TIER
	 + SLA checks     estimation            (configurable)
```

## Metrics

| Metric            | Description                        | Example Value |
|-------------------|------------------------------------|--------------|
| Cost per request  | $ spent per generation             | $0.0005      |
| SLA compliance    | % within cost/latency budget       | 98%          |
| Quality retention | % tasks matching baseline quality  | 92%          |
| $ Saved           | Savings vs. always using best      | $0.80 / 1.00 |

## Docker Deployment

You can run the API in a containerized environment using Docker:

### Build the Docker image
```bash
docker build -t cost-aware-router .
```

### Run the container
```bash
docker run -p 5000:5000 --env-file .env cost-aware-router
```

The API will be available at http://localhost:5000.

If you have Docker Compose, you can also use:
```bash
docker compose up --build
```


## How to run (CLI)

```bash
python -m cost_aware_router --prompt "Return a JSON object with fields 'name' and 'age' for Alice, age 30."
```

Or run the demo:

```bash
python scripts/adaptive_router_demo.py
```

## Evaluation

- Run the evaluation harness:
	```bash
	python eval/bench.py
	```
- Prompts and tasks in `eval/prompts.jsonl` (JSON, code, summarization)
- Metrics: cost, SLA, quality retention, and $ saved (see printed results)

## Training (original work)

- Model training scripts and configs are in `scripts/` and `configs/`
- Tokenizer and data prep: `scripts/train_tokenizer.py`, `scripts/prepare_data.py`
- Model code: `src/model.py`, training: `src/train.py`
- You can train your own CHEAP_TIER or QUALITY_TIER models and update `models/registry.yaml`

## Supported Models

The router supports multiple models for cost/quality/latency tradeoff and comparison:

- **from_scratch_125m** — Small, fast, from-scratch model (125M params)
- **teacher_gpt2_large** — Larger, higher-quality teacher model (e.g., GPT-2 Large)
- **gpt-3.5-turbo** — OpenAI GPT-3.5 Turbo (API)
- **gpt-4** — OpenAI GPT-4 (API)
- **llama-2-70b** — Meta Llama-2 70B (local or API)
- **mistral-8x7b** — Mistral 8x7B (local or API)

You can easily add or adjust models in `core/router.py` and `models/registry.yaml`.

---

**This system is designed to impress hiring managers and recruiters by focusing on measurable business value, not just ML training.**
