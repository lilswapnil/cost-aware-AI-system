
# Cost-Aware AI System — Adaptive LLM Router

This repository implements a **cost-aware AI system** that routes prompts to the most cost-efficient large language model (LLM) while balancing **quality**, **latency**, and **budget**. It is designed for real-world business outcomes, such as monthly spend tracking and maximizing savings versus a baseline model.

---

## Features

- **Adaptive LLM Routing:** Uses high-quality (expensive) models only when needed, falling back to cheaper models for low-complexity requests.
- **Cost, Latency, and Quality Constraints:** Optimizes for user-defined constraints on spend, response time, and output quality.
- **Spend Tracking:** Tracks monthly spend and estimated savings versus a baseline model.
- **Prompt Complexity Scoring:** Estimates prompt complexity to determine the minimum viable model.
- **Pluggable Model Registry:** Easily add or modify models and pricing in `models/registry.yaml` and `models/pricing.yaml`.
- **API and Demo:** REST API (FastAPI/Flask) and CLI demo for easy integration and experimentation.

---

## Project Structure

- `core/` — Core logic, model wrappers, and training scripts
- `cost_aware_router/` — Cost-aware router, adapters, API, caching, estimators, and routing policy
- `scripts/` — Demo, data preparation, and utility scripts
- `models/` — Model registry and pricing configuration
- `artifacts/` — Tokenizer and checkpoints
- `data/` — Training and validation data
- `docs/` — Documentation (e.g., training instructions)

---

## Installation

**Requirements:**
- Python 3.10+
- [PyTorch](https://pytorch.org/), [SentencePiece](https://github.com/google/sentencepiece), [FastAPI](https://fastapi.tiangolo.com/), [Flask](https://flask.palletsprojects.com/), [OpenAI Python SDK](https://github.com/openai/openai-python), and others (see `requirements.txt`)

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**(Optional) Docker:**

```bash
docker build -t cost-aware-router .
docker run -p 5000:5000 cost-aware-router
```

---

## Quick Start (Demo)

Run the adaptive router demo script:

```bash
python scripts/adaptive_router_demo.py
```

**Example output:**

```
Prompt: Summarize last week's support tickets into three bullet points.
-> Routed to local-7b | cost $0.0005 | latency 350ms

Prompt: Design a multi-step rollout plan and analyze trade-offs for risk mitigation.
-> Routed to gpt-4-class | cost $0.0108 | latency 1200ms

Total spend: 0.01 USD | Savings vs gpt-4-class: 0.01 USD
```

---

## API Usage

### FastAPI (Recommended)

Start the API server:

```bash
uvicorn cost_aware_router.api.main:app --reload
```

**POST /generate**

Request body (JSON):

```
{
	"prompt": "Your prompt here",
	"max_cost_usd": 0.01,
	"max_latency_ms": 1500,
	"min_quality": 0.7,
	"force_model": "cheap"  // or "openai" (optional)
}
```

Response:

```
{
	"text": "...",
	"chosen_model": "openai",
	"tokens_in": 42,
	"tokens_out": 120,
	"latency_ms": 1100,
	"estimated_cost_usd": 0.008,
	"quality_proxy": 0.82,
	"route_reason": "prompt_complexity_high(score=0.91)",
	"cache_hit": false,
	"saved_cost_usd": 0.002
}
```

### Flask (Legacy)

Start the Flask app:

```bash
python core/app.py
```

POST to `/route` with JSON:

```
{
	"prompt": "Your prompt here",
	"max_cost": 0.01,
	"max_latency": 1.5
}
```

---

## Training Your Own Model

See [docs/training.md](docs/training.md) for full instructions.

**Prerequisites:**
- SentencePiece tokenizer at `artifacts/tokenizer/llm_spm.model`
- Training data in `data/shards/train_*.npz` and validation in `data/shards/val_*.npz`

**Train:**

```bash
python core/train.py --config configs/125m.json
```

**Evaluate Perplexity:**

```bash
python core/eval.py --tok artifacts/tokenizer/llm_spm.model --ckpt artifacts/checkpoints/latest.pt --input data/sample.txt
```

---

## Model Registry & Pricing

- Add or edit models in `models/registry.yaml`
- Set pricing in `models/pricing.yaml`

---

## Contributing

Pull requests and issues are welcome! Please open an issue to discuss major changes first.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
