import math, torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

class RotaryEmbedding:
    # Simplified RoPE for demo
    def __init__(self, dim, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer = lambda name, buf: setattr(self, name, buf)
        self.register_buffer("inv_freq", inv_freq)

    def get_embed(self, T, device):
        t = torch.arange(T, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def apply_rotary(q, k, rope):
    # q, k: [B, H, T, D]
    T = q.size(2)
    emb = rope.get_embed(T, q.device)  # [T, D]
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    def rot(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).reshape_as(x)
    q = q * cos + rot(q) * sin
    k = k * cos + rot(k) * sin
    return q, k

class MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, rope_dim=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.d_head) if rope_dim else None

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3*d_model)
        qkv = qkv.view(B, T, self.n_heads, 3, self.d_head)  # (B, T, H, 3, D)
        qkv = qkv.permute(0, 2, 1, 3, 4)  # (B, H, T, 3, D)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]  # Each (B, H, T, D)
        if self.rope:
            q, k = apply_rotary(q, k, self.rope)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            att = att + attn_mask
        att = att.softmax(dim=-1)
        att = self.dropout(att)
        y = att @ v  # [B,H,T,D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MHA(d_model, n_heads, dropout=dropout, rope_dim=d_model//n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_len, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_dummy = nn.Parameter(torch.zeros(1))  # placeholder
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len
        self.register_buffer("mask", torch.triu(torch.ones(seq_len, seq_len)*float("-inf"), diagonal=1))

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.seq_len, "Sequence longer than model context"
        x = self.tok_emb(idx)
        attn_mask = self.mask[:T, :T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm(x)
        logits = self.head(x)
        return logits

def count_params(model):
    return sum(p.numel() for p in model.parameters())
