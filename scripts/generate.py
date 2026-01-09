import argparse, torch, sentencepiece as spm, yaml, time
from model import GPT
import os

def sample(model, idx, max_new_tokens, temperature=1.0, top_k=50):
    tokens_in = idx.shape[1]
    start_time = time.time()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.seq_len:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    tokens_out = idx.shape[1] - tokens_in
    elapsed = time.time() - start_time
    return idx, tokens_in, tokens_out, elapsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--tier", default="CHEAP_TIER", choices=["CHEAP_TIER", "QUALITY_TIER"])
    ap.add_argument("--max_cost", type=float, default=1.0)  # USD
    ap.add_argument("--max_latency", type=float, default=2.0)  # seconds
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tok)
    ck = torch.load(args.ckpt, map_location="cpu"); C = ck["config"]
    model = GPT(C["vocab_size"], C["d_model"], C["n_layers"], C["n_heads"], C["d_ff"], C["seq_len"], C["dropout"])
    model.load_state_dict(ck["model"]); model.eval()

    ids = [sp.bos_id()] + sp.EncodeAsIds(args.prompt)
    x = torch.tensor(ids).unsqueeze(0)

    import time
    start = time.perf_counter()
    out, tokens_in, tokens_out, elapsed = sample(model, x, args.max_new_tokens, temperature=0.8, top_k=50)
    latency = time.perf_counter() - start
    out = out[0].tolist()
    txt = sp.DecodeIds(out)

    # Load pricing table
    pricing_path = os.path.join(os.path.dirname(__file__), "../models/pricing.yaml")
    with open(pricing_path, "r") as f:
        pricing = yaml.safe_load(f)
    tier = args.tier
    price_in = pricing[tier]["price_in"]
    price_out = pricing[tier]["price_out"]
    cost_per_second = pricing[tier]["cost_per_second"]

    cost = (tokens_in / 1000 * price_in) + (tokens_out / 1000 * price_out) + (elapsed * cost_per_second)

    within_budget = cost <= args.max_cost
    within_latency = latency <= args.max_latency

    metadata = {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "elapsed_seconds": elapsed,
        "latency_seconds": latency,
        "latency_ms": latency * 1000,
        "cost_usd": cost,
        "tier": tier,
        "within_budget": within_budget,
        "within_latency": within_latency
    }
    print(txt)
    print("---METADATA---")
    print(metadata)

if __name__ == "__main__":
    main()
