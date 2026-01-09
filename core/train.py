import os, glob, json, math, time, argparse
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from model import GPT, count_params

class ShardDataset(IterableDataset):
    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
    def __iter__(self):
        rng = np.random.default_rng()
        while True:
            rng.shuffle(self.files)
            for f in self.files:
                batch = np.load(f)
                x, y = batch["x"], batch["y"]
                for i in range(len(x)):
                    yield torch.from_numpy(x[i].astype(np.int64)), torch.from_numpy(y[i].astype(np.int64))

def load_cfg(path):
    with open(path) as f: return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    C = load_cfg(args.config)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    model = GPT(
        vocab_size=C["vocab_size"],
        d_model=C["d_model"],
        n_layers=C["n_layers"],
        n_heads=C["n_heads"],
        d_ff=C["d_ff"],
        seq_len=C["seq_len"],
        dropout=C["dropout"]
    ).to(device)

    if C.get("bf16", False) and torch.cuda.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16))

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    print(f"Parameters: {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], betas=(0.9,0.95), weight_decay=C["weight_decay"])
    sched_warmup = C["warmup_steps"]
    max_steps = C["max_steps"]

    train_ds = ShardDataset(C["train_glob"])
    val_ds = ShardDataset(C["val_glob"])
    collate = lambda batch: (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
    )
    loader = DataLoader(train_ds, batch_size=C["micro_batch_size"], num_workers=0, collate_fn=collate)
    vloader = DataLoader(val_ds, batch_size=C["micro_batch_size"], num_workers=0, collate_fn=collate)

    grad_accum = C["grad_accum_steps"]
    log_every = C["log_every"]
    save_every = C["save_every"]
    save_dir = C["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    step = 0
    best_val = float("inf")
    loss_fn = nn.CrossEntropyLoss()

    def cosine_lr(step):
        if step < sched_warmup:
            return C["lr"] * (step+1) / sched_warmup
        progress = (step - sched_warmup) / max(1, (max_steps - sched_warmup))
        return 0.1 * C["lr"] + 0.9 * C["lr"] * (1 + math.cos(math.pi * progress)) / 2

    model.train()
    accum_loss = 0.0
    while step < max_steps:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dtype==torch.bfloat16)):
                logits = model(xb)
                loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
                loss = loss / grad_accum

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C["grad_clip"])

            if (step % grad_accum) == grad_accum - 1:
                for pg in opt.param_groups:
                    pg["lr"] = cosine_lr(step)
                opt.step()
                opt.zero_grad()

            accum_loss += loss.item()
            if step % log_every == 0 and step > 0 and rank == 0:
                print(f"step {step} | loss {accum_loss/log_every:.4f} | lr {opt.param_groups[0]['lr']:.2e}")
                accum_loss = 0.0

            if step % save_every == 0 and step > 0 and rank == 0:
                ckpt = {
                    "model": model.module.state_dict() if ddp else model.state_dict(),
                    "config": C,
                    "step": step
                }
                torch.save(ckpt, os.path.join(save_dir, f"step_{step}.pt"))
                torch.save(ckpt, os.path.join(save_dir, "latest.pt"))

            step += 1
            if step >= max_steps:
                break

        # quick val perplexity
        if rank == 0:
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                n = 8
                for _ in range(n):
                    xb, yb = next(iter(vloader))
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
                    vloss += loss.item()
                vloss /= n
                ppl = math.exp(min(20, vloss))
                print(f"[val] step {step} | loss {vloss:.4f} | ppl {ppl:.2f}")
                if vloss < best_val:
                    best_val = vloss
                    torch.save({"model": model.module.state_dict() if ddp else model.state_dict(),
                                "config": C, "step": step},
                               os.path.join(save_dir, "best.pt"))
            model.train()

    if ddp:
        torch.distributed.destroy_process_group()

    if rank == 0:
        print("Training complete. Latest checkpoint saved at:", os.path.join(save_dir, "latest.pt"))

if __name__ == "__main__":
    main()
