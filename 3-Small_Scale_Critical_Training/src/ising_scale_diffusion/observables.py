from __future__ import annotations

import torch


def _as_spins(spins: torch.Tensor) -> torch.Tensor:
    values = spins.float()
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError("spins must have shape [H, W] or [B, H, W]")
    if torch.any((values != -1) & (values != 1)):
        raise ValueError("spins must contain only -1 and +1")
    return values


def energy_density(spins: torch.Tensor, periodic: bool = False) -> torch.Tensor:
    values = _as_spins(spins)
    if periodic:
        horizontal = values * torch.roll(values, shifts=-1, dims=2)
        vertical = values * torch.roll(values, shifts=-1, dims=1)
        return -(horizontal.sum(dim=(1, 2)) + vertical.sum(dim=(1, 2))) / (
            values.shape[1] * values.shape[2]
        )
    horizontal = values[:, :, :-1] * values[:, :, 1:]
    vertical = values[:, :-1, :] * values[:, 1:, :]
    return -(horizontal.sum(dim=(1, 2)) + vertical.sum(dim=(1, 2))) / (
        values.shape[1] * values.shape[2]
    )


def magnetization(spins: torch.Tensor) -> torch.Tensor:
    return _as_spins(spins).mean(dim=(1, 2))


def binder_cumulant(spins: torch.Tensor) -> torch.Tensor:
    moments = magnetization(spins)
    mean_m2 = moments.square().mean()
    mean_m4 = moments.pow(4).mean()
    return 1.0 - mean_m4 / (3.0 * mean_m2.square().clamp_min(1e-12))


def open_axial_correlation(spins: torch.Tensor, max_distance: int) -> torch.Tensor:
    values = _as_spins(spins)
    maximum = min(max_distance, values.shape[1] - 1, values.shape[2] - 1)
    correlations = [torch.ones((), dtype=values.dtype, device=values.device)]
    for distance in range(1, maximum + 1):
        horizontal = (values[:, :, :-distance] * values[:, :, distance:]).mean()
        vertical = (values[:, :-distance, :] * values[:, distance:, :]).mean()
        correlations.append(0.5 * (horizontal + vertical))
    return torch.stack(correlations)


def summarize(spins: torch.Tensor, max_distance: int = 16) -> dict[str, object]:
    values = _as_spins(spins)
    m = magnetization(values)
    return {
        "n_samples": values.shape[0],
        "height": values.shape[1],
        "width": values.shape[2],
        "mean_energy_open": float(energy_density(values).mean().item()),
        "mean_magnetization": float(m.mean().item()),
        "mean_abs_magnetization": float(m.abs().mean().item()),
        "binder_u4": float(binder_cumulant(values).item()),
        "open_axial_correlation": open_axial_correlation(values, max_distance)
        .cpu()
        .tolist(),
    }
