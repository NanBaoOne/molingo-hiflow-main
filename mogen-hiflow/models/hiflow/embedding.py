import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
        t = t * time_factor
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


def rope_frequencies(position_ids, dim, theta=10000.0):
    if dim % 2 != 0:
        raise ValueError("RoPE head dimension must be even")
    pos = position_ids.squeeze(-1).float()
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device).float() / dim))
    freqs = torch.einsum("bt,d->btd", pos, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x_float = x.float()
    x1 = x_float[..., 0::2]
    x2 = x_float[..., 1::2]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(dtype=x.dtype)


def lengths_to_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
