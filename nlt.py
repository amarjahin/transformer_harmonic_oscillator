import math 
import torch 
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
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
        # mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        # scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, nH, T, T)
        out = attn @ v                    # (B, nH, T, dH)

        # Merge heads: (B, nH, T, dH) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        out = self.Wo(out)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int = None):
        super().__init__()
        d_mlp = d_mlp or 4 * d_model
        self.up = nn.Linear(d_model, d_mlp, bias=True)
        self.down = nn.Linear(d_mlp, d_model, bias=True)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class nlt(nn.Module):
    def __init__(self, d_model:int, d_head:int, n_layers:int, d_mlp:int=None, use_mlp:bool=True):
        super().__init__()
        self.embed = nn.Linear(2, d_model, bias=True)
        self.use_mlp = use_mlp
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict({"attn": Attention(d_model=d_model, d_head=d_head)})
            if use_mlp:
                block["mlp"] = MLP(d_model, d_mlp)
            self.blocks.append(block)
        self.unembed = nn.Linear(d_model, 2, bias=True)  # last-token -> next (x,p)

    def forward(self, x):
        # x: (B, T, 2)
        h = self.embed(x)     # (B, T, d_model)
        for block in self.blocks:
            h = h + block["attn"](h)
            if self.use_mlp:
                h = h + block["mlp"](h)
        y = self.unembed(h)   # (B, T, 2)
        return y


def get_attention_weights(model, x_in, layer_idx=-1):
    """
    Recompute attention weights from model parameters (no model modification).
    Returns attn (B, nH, T, T) for the specified layer.
    """
    h = model.embed(x_in)
    target = layer_idx if layer_idx >= 0 else len(model.blocks) + layer_idx
    for i, block in enumerate(model.blocks):
        if i == target:
            attn_module = block["attn"]
            B, T, D = h.shape
            qkv = attn_module.Wqkv(h)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, attn_module.n_heads, attn_module.d_head).transpose(1, 2)
            k = k.view(B, T, attn_module.n_heads, attn_module.d_head).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn_module.d_head)
            mask = torch.tril(torch.ones(T, T, device=h.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            return F.softmax(scores, dim=-1)
        h = h + block["attn"](h)
        if model.use_mlp:
            h = h + block["mlp"](h)
    return None

