import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TimestepEmbedder, apply_rope, rope_frequencies


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, position_ids=None, key_padding_mask=None):
        device = x.device
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if position_ids is not None:
            position_ids = position_ids.to(device)
            cos, sin = rope_frequencies(position_ids, self.head_dim)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_mask = None
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device)
            attn_mask = (~key_padding_mask)[:, None, None, :]
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        x = x.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.proj(x)


class DoubleStreamBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.motion_norm1 = RMSNorm(dim)
        self.text_norm1 = RMSNorm(dim)
        self.motion_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.text_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.motion_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.text_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.motion_mlp = MLP(dim, mlp_ratio)
        self.text_mlp = MLP(dim, mlp_ratio)
        self.motion_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.text_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def _stream(self, x, c, norm1, attn, norm2, mlp, mod, position_ids, key_padding_mask):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod(c).chunk(6, dim=-1)
        h = modulate(norm1(x), shift_a, scale_a)
        x = x + gate_a[:, None, :] * attn(h, position_ids=position_ids, key_padding_mask=key_padding_mask)
        h = modulate(norm2(x), shift_m, scale_m)
        x = x + gate_m[:, None, :] * mlp(h)
        return x

    def forward(self, motion, text, c, motion_ids, text_ids, motion_mask=None, text_mask=None):
        motion = self._stream(motion, c, self.motion_norm1, self.motion_attn, self.motion_norm2,
                              self.motion_mlp, self.motion_mod, motion_ids, motion_mask)
        text = self._stream(text, c, self.text_norm1, self.text_attn, self.text_norm2,
                            self.text_mlp, self.text_mod, text_ids, text_mask)
        return motion, text


class SingleStreamBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, mlp_ratio)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, position_ids, key_padding_mask=None):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.mod(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_a, scale_a)
        x = x + gate_a[:, None, :] * self.attn(h, position_ids=position_ids, key_padding_mask=key_padding_mask)
        h = modulate(self.norm2(x), shift_m, scale_m)
        return x + gate_m[:, None, :] * self.mlp(h)


class MotionFlux(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, depth=16, num_heads=16,
                 mlp_ratio=4.0, double_depth=None, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.x_embedder = nn.Linear(input_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        double_depth = depth // 2 if double_depth is None else double_depth
        single_depth = depth - double_depth
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(double_depth)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(single_depth)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.final = nn.Linear(hidden_size, output_size)

    def forward(self, x, timesteps, text, pooled, motion_ids, text_ids, motion_mask=None, text_mask=None):
        device = x.device
        timesteps = timesteps.to(device)
        text = text.to(device)
        pooled = pooled.to(device)
        motion_ids = motion_ids.to(device)
        text_ids = text_ids.to(device)
        if motion_mask is not None:
            motion_mask = motion_mask.to(device)
        if text_mask is not None:
            text_mask = text_mask.to(device)
        x = self.x_embedder(x)
        text = self.text_proj(text)
        c = self.t_embedder(timesteps) + pooled
        for block in self.double_blocks:
            x, text = block(x, text, c, motion_ids, text_ids, motion_mask=motion_mask, text_mask=text_mask)
        joint = torch.cat([text, x], dim=1)
        ids = torch.cat([text_ids, motion_ids], dim=1)
        if text_mask is not None or motion_mask is not None:
            text_padding = torch.zeros(joint.shape[0], text.shape[1], dtype=torch.bool, device=joint.device)
            motion_padding = torch.zeros(joint.shape[0], x.shape[1], dtype=torch.bool, device=joint.device)
            if text_mask is not None:
                text_padding = text_mask
            if motion_mask is not None:
                motion_padding = motion_mask
            key_padding_mask = torch.cat([text_padding, motion_padding], dim=1)
        else:
            key_padding_mask = None
        for block in self.single_blocks:
            joint = block(joint, c, ids, key_padding_mask=key_padding_mask)
        x = joint[:, text.shape[1]:]
        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.final(x)
