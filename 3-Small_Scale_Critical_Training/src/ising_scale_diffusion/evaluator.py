from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from .data import FixedCropSource, ReplayableCropSource
from .system import IsingDiffusionSystem


@dataclass(frozen=True)
class ValidationResult:
    nelbo: float
    rows: list[dict[str, float | int]]


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_fixed_nelbo(
    system: IsingDiffusionSystem,
    source: FixedCropSource | ReplayableCropSource,
    widths: tuple[int, ...],
    t_grid: tuple[float, ...],
    batches_per_width: int,
    device: torch.device,
    autocast_factory: Callable[[], AbstractContextManager],
    step: int,
) -> ValidationResult:
    rows: list[dict[str, float | int]] = []
    system.eval()
    for width in widths:
        if isinstance(source, FixedCropSource) and width not in source.widths:
            raise ValueError(f"Validation manifest has no crops at width {width}")
        for t_index, diffusion_t in enumerate(t_grid):
            for batch_index in range(batches_per_width):
                if isinstance(source, FixedCropSource):
                    batch = source.batch(width, batch_index, t_index)
                else:
                    batch = source.batch(
                        step=batch_index,
                        width=width,
                        namespace=f"validation-t{t_index}",
                    )
                batch = _move_batch(batch, device)
                with autocast_factory():
                    result = system.loss_with_metrics(batch, diffusion_t=diffusion_t)
                rows.append(
                    {
                        "step": step,
                        "width": width,
                        "diffusion_t": diffusion_t,
                        "nelbo": float(result.loss.item()),
                        "masked_ce": float(result.masked_ce.item()),
                        "masked_accuracy": float(result.masked_accuracy.item()),
                        "mask_fraction": float(result.mask_fraction.item()),
                    }
                )
    system.train()
    return ValidationResult(
        nelbo=sum(float(row["nelbo"]) for row in rows) / len(rows), rows=rows
    )
