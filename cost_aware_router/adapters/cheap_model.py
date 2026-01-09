"""Adapter for the local from-scratch GPT model."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import sentencepiece as spm
import torch

from cost_aware_router.adapters.local_model import GPT


@dataclass
class CheapGeneration:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    avg_entropy: Optional[float]


class CheapModelAdapter:
    def __init__(
        self,
        tokenizer_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        self.tokenizer_path = tokenizer_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self._model = self._load_model(checkpoint_path)
        self._model.to(self.device)
        self._model.eval()

    def _load_model(self, checkpoint_path: str) -> GPT:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = ckpt["config"]
        model = GPT(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            seq_len=config["seq_len"],
            dropout=config.get("dropout", 0.0),
        )
        model.load_state_dict(ckpt["model"])
        return model

    def _encode(self, prompt: str) -> List[int]:
        return [self._tokenizer.bos_id()] + self._tokenizer.EncodeAsIds(prompt)

    def _decode(self, tokens: List[int]) -> str:
        return self._tokenizer.DecodeIds(tokens)

    def generate(self, prompt: str) -> CheapGeneration:
        start = time.monotonic()
        input_ids = self._encode(prompt)
        tokens = input_ids[:]
        entropies: List[float] = []
        for _ in range(self.max_new_tokens):
            idx = torch.tensor(tokens[-self._model.seq_len :], device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self._model(idx)
            next_logits = logits[:, -1, :]
            if self.temperature > 0:
                next_logits = next_logits / self.temperature
            probs = torch.softmax(next_logits, dim=-1)
            entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=-1)
            entropies.append(float(entropy.item()))
            next_id = int(torch.argmax(probs, dim=-1).item())
            tokens.append(next_id)
            if next_id == self._tokenizer.eos_id():
                break
        text = self._decode(tokens[len(input_ids) :])
        latency_ms = int((time.monotonic() - start) * 1000)
        avg_entropy = sum(entropies) / len(entropies) if entropies else None
        return CheapGeneration(
            text=text,
            tokens_in=len(input_ids),
            tokens_out=max(len(tokens) - len(input_ids), 0),
            latency_ms=latency_ms,
            avg_entropy=avg_entropy,
        )
