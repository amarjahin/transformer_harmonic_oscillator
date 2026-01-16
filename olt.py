import math 
import torch 
from torch import nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    def __init__(self, d_model:int, d_head:int):
        super().__init__()
        if d_model % d_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by d_head ({d_head}).")

        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = d_model // d_head

        # Project to all heads at once: (B,T,d_model) -> (B,T,3*d_model)
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection back to d_model
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, D = x.shape
        assert D == self.d_model
        qkv = self.Wqkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, D)
        # Reshape into heads: (B, T, D) -> (B, nH, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nH, T, dH)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nH, T, dH)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nH, T, dH)

         # Attention scores: (B, nH, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal mask (broadcasts over B and heads)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, nH, T, T)
        out = attn @ v                    # (B, nH, T, dH)

        # Merge heads: (B, nH, T, dH) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        out = self.Wo(out) 
        return out



class olt(nn.Module):
    def __init__(self, d_model:int, d_head:int):
        super().__init__()
        self.embed = nn.Linear(2, d_model, bias=True)
        self.attn = CausalAttention(d_model=d_model, d_head=d_head)
        self.unembed = nn.Linear(d_model, 2, bias=True)  # last-token -> next (x,p)

    def forward(self, x):
        # x: (B, T, 2)
        h = self.embed(x)     # (B, T, d_model)

        h = h + self.attn(h)      # (B, T, d_model)
        y = self.unembed(h[:, -1, :])  # (B, 2)
        return y
