from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


class FourierFeatures(nn.Module):
    def __init__(self, n_frequencies: int, max_frequency: float = 1_000.0) -> None:
        super().__init__()
        frequencies = torch.logspace(
            0.0, math.log10(max_frequency), steps=n_frequencies
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.frequencies.numel()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float().reshape(-1, 1)
        angles = 2.0 * math.pi * values * self.frequencies.reshape(1, -1)
        return torch.cat((values, torch.sin(angles), torch.cos(angles)), dim=-1)


class ConditionEncoder(nn.Module):
    def __init__(self, dim: int, condition_on_beta: bool, fixed_beta: float) -> None:
        super().__init__()
        n_frequencies = max(4, dim // 16)
        self.time_features = FourierFeatures(n_frequencies)
        self.beta_features = FourierFeatures(n_frequencies)
        self.condition_on_beta = condition_on_beta
        self.fixed_beta = float(fixed_beta)
        feature_dim = self.time_features.output_dim
        if condition_on_beta:
            feature_dim += self.beta_features.output_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self, diffusion_t: torch.Tensor, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = [self.time_features(diffusion_t)]
        if self.condition_on_beta:
            if beta is None:
                raise ValueError("beta is required when condition_on_beta=True")
            normalized_beta = (beta.float() - self.fixed_beta) / self.fixed_beta
            features.append(self.beta_features(normalized_beta))
        return self.net(torch.cat(features, dim=-1))


def _head_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * scale.to(dtype=x.dtype)


def _apply_rope(x: torch.Tensor, base: float) -> torch.Tensor:
    """Apply one-dimensional RoPE to [batch, heads, length, head_dim]."""
    head_dim = x.shape[-1]
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    exponent = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
    inverse_frequency = base ** (-exponent / head_dim)
    angles = positions[:, None] * inverse_frequency[None, :]
    cosine = angles.cos().to(dtype=x.dtype)[None, None, :, :]
    sine = angles.sin().to(dtype=x.dtype)[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    )
    return rotated.flatten(-2)


class SharedAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if (dim // heads) % 2:
            raise ValueError("head dimension must be even")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.qk_norm = qk_norm
        self.rope_base = float(rope_base)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if self.qk_norm:
            q, k = _head_rms_norm(q), _head_rms_norm(k)
        q, k = _apply_rope(q, self.rope_base), _apply_rope(k, self.rope_base)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = x.shape
        qkv = self.qkv(x).reshape(batch, height, width, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)

        row_q = q.permute(0, 1, 3, 2, 4).reshape(
            batch * height, self.heads, width, self.head_dim
        )
        row_k = k.permute(0, 1, 3, 2, 4).reshape_as(row_q)
        row_v = v.permute(0, 1, 3, 2, 4).reshape_as(row_q)
        row = self._attention(row_q, row_k, row_v)
        row = row.reshape(batch, height, self.heads, width, self.head_dim)
        row = row.permute(0, 1, 3, 2, 4).reshape(batch, height, width, self.dim)

        column_q = q.permute(0, 2, 3, 1, 4).reshape(
            batch * width, self.heads, height, self.head_dim
        )
        column_k = k.permute(0, 2, 3, 1, 4).reshape_as(column_q)
        column_v = v.permute(0, 2, 3, 1, 4).reshape_as(column_q)
        column = self._attention(column_q, column_k, column_v)
        column = column.reshape(batch, width, self.heads, height, self.head_dim)
        column = column.permute(0, 3, 1, 2, 4).reshape(batch, height, width, self.dim)
        return self.out((row + column) / math.sqrt(2.0))


class ConditionedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        qk_norm: bool,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attention = SharedAxialAttention(
            dim, heads, dropout=dropout, qk_norm=qk_norm, rope_base=rope_base
        )
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(
        normalized: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return normalized * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            condition
        ).chunk(6, dim=-1)
        attention_input = self._modulate(self.attention_norm(x), shift_a, scale_a)
        x = x + gate_a[:, None, None, :] * self.attention(attention_input)
        mlp_input = self._modulate(self.mlp_norm(x), shift_m, scale_m)
        return x + gate_m[:, None, None, :] * self.mlp(mlp_input)


class IsingDiffusionModel(nn.Module):
    """Bidirectional conditional Transformer over a two-dimensional spin field."""

    mask_token = 2

    def __init__(
        self,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_base: float = 10_000.0,
        condition_on_beta: bool = False,
        fixed_beta: float = 0.44068679350977147,
    ) -> None:
        super().__init__()
        self.condition_on_beta = condition_on_beta
        self.fixed_beta = float(fixed_beta)
        self.token_embedding = nn.Embedding(3, dim)
        self.condition = ConditionEncoder(dim, condition_on_beta, fixed_beta)
        self.condition_to_input = nn.Linear(dim, dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                ConditionedBlock(dim, heads, mlp_ratio, dropout, qk_norm, rope_base)
                for _ in range(depth)
            ]
        )
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, 2, bias=False)

    def forward(
        self,
        noisy_tokens: torch.Tensor,
        diffusion_t: torch.Tensor,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_tokens.ndim != 3:
            raise ValueError("noisy_tokens must have shape [B, H, W]")
        if noisy_tokens.dtype != torch.long:
            raise TypeError("noisy_tokens must use torch.int64")
        if torch.any((noisy_tokens < 0) | (noisy_tokens > self.mask_token)):
            raise ValueError("noisy_tokens values must be 0, 1, or MASK=2")
        if diffusion_t.shape != (noisy_tokens.shape[0],):
            raise ValueError("diffusion_t must have shape [B]")
        condition = self.condition(diffusion_t, beta)
        hidden = self.token_embedding(noisy_tokens)
        hidden = hidden + self.condition_to_input(condition)[:, None, None, :]
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.head(self.final_norm(hidden))

    def architecture_facts(self) -> dict[str, int | float | bool | str]:
        first = self.blocks[0].attention
        return {
            "architecture": "shared-row-column-axial-transformer",
            "input_vocabulary": 3,
            "output_vocabulary": 2,
            "dim": first.dim,
            "depth": len(self.blocks),
            "heads": first.heads,
            "head_dim": first.head_dim,
            "rope_base": first.rope_base,
            "learned_absolute_position": False,
            "condition_on_beta": self.condition_on_beta,
            "fixed_beta": self.fixed_beta,
        }
