# Training the Local Model

This repository includes a from-scratch GPT implementation in `src/model.py` and a training loop in `src/train.py`.

## Prerequisites

- A SentencePiece tokenizer at `artifacts/tokenizer/llm_spm.model`.
- Training data shards at `data/shards/train_*.npz` and validation shards at `data/shards/val_*.npz`.

## Train

```bash
python src/train.py --config configs/125m.json
```

Checkpoints are written to `artifacts/checkpoints/` (including `latest.pt`). Use that checkpoint when running the cost-aware router.

## Evaluate Perplexity

```bash
python src/eval.py --tok artifacts/tokenizer/llm_spm.model --ckpt artifacts/checkpoints/latest.pt --input data/sample.txt
```
