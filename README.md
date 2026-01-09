# Cost-Aware AI System — Adaptive LLM Router

This repository now ships a production-style **Cost-Aware AI System** that routes prompts between:

- **CHEAP_TIER**: the from-scratch local GPT model in this repo.
- **QUALITY_TIER**: OpenAI Responses API for high-quality inference.

The router makes decisions using **max_cost_usd**, **max_latency_ms**, **min_quality**, and the **cheap model’s uncertainty**. It only escalates to OpenAI when required and always returns detailed routing metadata.

## What’s Included

- **`cost_aware_router/` package** with CLI + FastAPI server.
- **Exact prompt caching** using SQLite.
- **Routing policy** based on prompt complexity and cheap-model uncertainty.
- **Quality proxy scoring** (JSON validity, constraints, repetition penalty, structured output bonus).
- **Evaluation harness** comparing cheap-only, openai-only, and routed runs.
- **Pricing configuration** for cost estimation.

## How to Run

### 1) Set your OpenAI key

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
```

### 2) CLI Usage

```bash
python -m cost_aware_router \
  --prompt "Summarize the benefits of test-driven development." \
  --min-quality 0.6 \
  --max-cost-usd 0.02
```

### 3) FastAPI Server

```bash
uvicorn cost_aware_router.api.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Return JSON with keys: summary, risks", "min_quality": 0.7}'
```

### 4) Evaluation Harness

```bash
python -m cost_aware_router.eval.bench
```

## Configuration

Defaults live in `cost_aware_router/configs/default.yaml` and include:

- Cheap model tokenizer + checkpoint paths.
- OpenAI model name (default `gpt-4o-mini`).
- Routing thresholds and cache path.

## Training the Local Model

See [`docs/training.md`](docs/training.md) for training and evaluation instructions.

## Outputs

Each generation returns:

- `text`
- `chosen_model`: `cheap` or `openai`
- `tokens_in`, `tokens_out`
- `latency_ms`
- `estimated_cost_usd`
- `quality_proxy` (0..1)
- `route_reason`
- `cache_hit`
- `saved_cost_usd`
