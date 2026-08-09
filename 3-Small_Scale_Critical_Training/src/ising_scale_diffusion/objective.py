from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Corruption:
    noisy_tokens: torch.Tensor
    mask: torch.Tensor
    diffusion_t: torch.Tensor


@dataclass(frozen=True)
class LossResult:
    loss: torch.Tensor
    masked_ce: torch.Tensor
    masked_accuracy: torch.Tensor
    mask_fraction: torch.Tensor
    diffusion_t_mean: torch.Tensor


class AbsorbingDiffusionObjective:
    mask_token = 2

    def __init__(self, t_min: float = 0.2, t_max: float = 1.0) -> None:
        if not 0.0 < t_min <= t_max <= 1.0:
            raise ValueError("Expected 0 < t_min <= t_max <= 1")
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def corrupt(
        self,
        clean_tokens: torch.Tensor,
        noise_seeds: torch.Tensor,
        diffusion_t: torch.Tensor | float | None = None,
    ) -> Corruption:
        if clean_tokens.ndim != 3 or clean_tokens.dtype != torch.long:
            raise TypeError("clean_tokens must be int64 [B, H, W]")
        if torch.any((clean_tokens < 0) | (clean_tokens > 1)):
            raise ValueError("clean_tokens must contain only 0 or 1")
        batch, height, width = clean_tokens.shape
        if noise_seeds.shape != (batch,):
            raise ValueError("noise_seeds must have shape [B]")

        if diffusion_t is None:
            fixed_times: torch.Tensor | None = None
        elif isinstance(diffusion_t, torch.Tensor):
            fixed_times = diffusion_t.detach().float().cpu().reshape(-1)
            if fixed_times.numel() == 1:
                fixed_times = fixed_times.expand(batch)
            if fixed_times.shape != (batch,):
                raise ValueError("diffusion_t must be scalar or shape [B]")
        else:
            fixed_times = torch.full((batch,), float(diffusion_t), dtype=torch.float32)

        times: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for index, raw_seed in enumerate(noise_seeds.detach().cpu().tolist()):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(raw_seed))
            if fixed_times is None:
                unit = torch.rand((), generator=generator)
                current_t = self.t_min + (self.t_max - self.t_min) * unit
            else:
                current_t = fixed_times[index]
            if not self.t_min <= float(current_t) <= self.t_max:
                raise ValueError("diffusion_t lies outside configured bounds")
            mask = torch.rand((height, width), generator=generator) < current_t
            times.append(current_t.reshape(()))
            masks.append(mask)

        time_tensor = torch.stack(times).to(clean_tokens.device)
        mask_tensor = torch.stack(masks).to(clean_tokens.device)
        noisy = torch.where(mask_tensor, self.mask_token, clean_tokens)
        return Corruption(noisy, mask_tensor, time_tensor)

    @staticmethod
    def loss_from_logits(
        logits: torch.Tensor,
        clean_tokens: torch.Tensor,
        mask: torch.Tensor,
        diffusion_t: torch.Tensor,
    ) -> LossResult:
        if logits.shape != (*clean_tokens.shape, 2):
            raise ValueError("logits must have shape [B, H, W, 2]")
        batch, height, width = clean_tokens.shape
        per_site_ce = F.cross_entropy(
            logits.float().reshape(-1, 2),
            clean_tokens.reshape(-1),
            reduction="none",
        ).reshape(batch, height, width)
        mask_float = mask.float()
        weighted = per_site_ce * mask_float / diffusion_t[:, None, None].float()
        per_sample = weighted.sum(dim=(1, 2)) / float(height * width)
        loss = per_sample.mean()

        masked_count = mask_float.sum()
        safe_count = masked_count.clamp_min(1.0)
        masked_ce = (per_site_ce * mask_float).sum() / safe_count
        predictions = logits.argmax(dim=-1)
        masked_accuracy = (
            (predictions == clean_tokens) & mask
        ).float().sum() / safe_count
        zero = loss.detach() * 0.0
        masked_ce = torch.where(masked_count > 0, masked_ce, zero)
        masked_accuracy = torch.where(masked_count > 0, masked_accuracy, zero)
        return LossResult(
            loss=loss,
            masked_ce=masked_ce,
            masked_accuracy=masked_accuracy,
            mask_fraction=mask_float.mean(),
            diffusion_t_mean=diffusion_t.float().mean(),
        )
