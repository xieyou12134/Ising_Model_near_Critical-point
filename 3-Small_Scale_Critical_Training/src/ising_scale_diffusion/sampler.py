from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import IsingDiffusionModel


@dataclass(frozen=True)
class SamplingResult:
    tokens: torch.Tensor
    spins: torch.Tensor
    trace: list[dict[str, float | int]]


def cosine_time_grid(steps: int, device: torch.device | str = "cpu") -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    phase = torch.linspace(0.0, torch.pi / 2.0, steps + 1, device=device)
    times = torch.cos(phase).square()
    times[0], times[-1] = 1.0, 0.0
    if not torch.all(times[:-1] > times[1:]):
        raise RuntimeError("Reverse time grid must be strictly decreasing")
    return times


@torch.no_grad()
def sample_absorbing(
    model: IsingDiffusionModel,
    shape: tuple[int, int, int],
    beta: float | torch.Tensor | None = None,
    steps: int = 64,
    sampling_temperature: float = 1.0,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> SamplingResult:
    if sampling_temperature <= 0:
        raise ValueError("sampling_temperature must be positive")
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        raise ValueError("shape must be (batch, height, width) with positive values")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    batch, height, width = shape
    if model.condition_on_beta and beta is None:
        raise ValueError("beta is required for a beta-conditioned model")

    if beta is None:
        beta_tensor = None
    elif isinstance(beta, torch.Tensor):
        beta_tensor = beta.to(device=device, dtype=torch.float32).reshape(-1)
        if beta_tensor.numel() == 1:
            beta_tensor = beta_tensor.expand(batch)
        if beta_tensor.shape != (batch,):
            raise ValueError("beta tensor must be scalar or shape [B]")
    else:
        beta_tensor = torch.full((batch,), float(beta), device=device)

    tokens = torch.full(
        (batch, height, width), model.mask_token, device=device, dtype=torch.long
    )
    times = cosine_time_grid(steps, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    trace: list[dict[str, float | int]] = []

    was_training = model.training
    model.eval()
    try:
        for index in range(steps):
            current_t = times[index]
            next_t = times[index + 1]
            time_batch = current_t.expand(batch)
            model_beta = beta_tensor if model.condition_on_beta else None
            logits = model(tokens, time_batch, beta=model_beta)
            probabilities = torch.softmax(logits.float() / sampling_temperature, dim=-1)
            proposals = torch.multinomial(
                probabilities.reshape(-1, 2), 1, generator=generator
            ).reshape(batch, height, width)
            masked = tokens == model.mask_token
            reveal_probability = 1.0 - next_t / current_t
            if index == steps - 1:
                reveal = masked
            else:
                reveal = masked & (
                    torch.rand(tokens.shape, device=device, generator=generator)
                    < reveal_probability
                )
            tokens = torch.where(reveal, proposals, tokens)
            masked_probabilities = probabilities[masked]
            if masked_probabilities.numel():
                entropy = (
                    -(
                        masked_probabilities
                        * masked_probabilities.clamp_min(1e-12).log()
                    )
                    .sum(dim=-1)
                    .mean()
                )
                entropy_value = float(entropy.item())
            else:
                entropy_value = 0.0
            trace.append(
                {
                    "step": index + 1,
                    "current_t": float(current_t.item()),
                    "next_t": float(next_t.item()),
                    "revealed": int(reveal.sum().item()),
                    "remaining_mask_fraction": float(
                        (tokens == model.mask_token).float().mean().item()
                    ),
                    "mean_entropy": entropy_value,
                }
            )
    finally:
        model.train(was_training)

    if torch.any(tokens == model.mask_token):
        raise RuntimeError("Absorbing sampler finished with masked sites")
    return SamplingResult(tokens=tokens, spins=tokens * 2 - 1, trace=trace)
