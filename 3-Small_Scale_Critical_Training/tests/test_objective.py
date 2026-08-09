from __future__ import annotations

import math

import pytest
import torch

from ising_scale_diffusion.model import IsingDiffusionModel
from ising_scale_diffusion.objective import AbsorbingDiffusionObjective
from ising_scale_diffusion.system import IsingDiffusionSystem


def test_t_one_masks_every_site_and_is_replayable() -> None:
    objective = AbsorbingDiffusionObjective()
    clean = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    seeds = torch.tensor([123], dtype=torch.long)
    first = objective.corrupt(clean, seeds, diffusion_t=1.0)
    second = objective.corrupt(clean, seeds, diffusion_t=1.0)
    assert torch.all(first.mask)
    assert torch.all(first.noisy_tokens == objective.mask_token)
    assert torch.equal(first.noisy_tokens, second.noisy_tokens)


def test_fixed_site_nelbo_and_empty_mask() -> None:
    objective = AbsorbingDiffusionObjective()
    clean = torch.zeros((2, 3, 3), dtype=torch.long)
    logits = torch.zeros((2, 3, 3, 2))
    full = objective.loss_from_logits(
        logits, clean, torch.ones_like(clean, dtype=torch.bool), torch.ones(2)
    )
    assert full.loss.item() == pytest.approx(math.log(2.0))
    empty = objective.loss_from_logits(
        logits, clean, torch.zeros_like(clean, dtype=torch.bool), torch.ones(2)
    )
    assert empty.loss.item() == 0.0
    assert empty.masked_ce.item() == 0.0
    assert torch.isfinite(empty.loss)


def test_fixed_beta_system_rejects_other_temperatures() -> None:
    model = IsingDiffusionModel(dim=32, depth=1, heads=4, condition_on_beta=False)
    system = IsingDiffusionSystem(model, AbsorbingDiffusionObjective())
    batch = {
        "clean_tokens": torch.zeros((1, 4, 4), dtype=torch.long),
        "beta": torch.tensor([0.3]),
        "noise_seeds": torch.tensor([1]),
    }
    with pytest.raises(ValueError, match="fixes beta"):
        system.loss(batch)
