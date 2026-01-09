import argparse, math, torch, sentencepiece as spm
from model import GPT

def load_ckpt(path):
    return torch.load(path, map_location="cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tok)
    text = open(args.input, "r", encoding="utf-8").read()
    ids = [sp.bos_id()] + sp.EncodeAsIds(text) + [sp.eos_id()]

    ck = load_ckpt(args.ckpt)
    C = ck["config"]
    model = GPT(
        vocab_size=C["vocab_size"], d_model=C["d_model"], n_layers=C["n_layers"],
        n_heads=C["n_heads"], d_ff=C["d_ff"], seq_len=C["seq_len"], dropout=C["dropout"]
    )
    model.load_state_dict(ck["model"])
    model.eval()

    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    for i in range(0, len(ids)-args.seq_len-1, args.seq_len):
        x = torch.tensor(ids[i:i+args.seq_len]).unsqueeze(0)
        y = torch.tensor(ids[i+1:i+1+args.seq_len]).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            n += 1
    avg = total_loss / max(1, n)
    ppl = math.exp(min(20, avg))
    print(f"Val loss: {avg:.4f} | Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()
