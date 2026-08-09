from __future__ import annotations

from itertools import pairwise

import torch

from ising_scale_diffusion.model import IsingDiffusionModel
from ising_scale_diffusion.sampler import cosine_time_grid, sample_absorbing


def test_time_grid_is_strict_and_complete() -> None:
    grid = cosine_time_grid(8)
    assert grid[0].item() == 1.0
    assert grid[-1].item() == 0.0
    assert torch.all(grid[:-1] > grid[1:])


def test_sampler_is_reproducible_and_removes_all_masks() -> None:
    torch.manual_seed(9)
    model = IsingDiffusionModel(dim=32, depth=1, heads=4).eval()
    first = sample_absorbing(model, (2, 6, 6), steps=5, seed=88)
    second = sample_absorbing(model, (2, 6, 6), steps=5, seed=88)
    assert torch.equal(first.tokens, second.tokens)
    assert set(torch.unique(first.spins).tolist()) <= {-1, 1}
    remaining = [row["remaining_mask_fraction"] for row in first.trace]
    assert all(left >= right for left, right in pairwise(remaining))
    assert remaining[-1] == 0.0
