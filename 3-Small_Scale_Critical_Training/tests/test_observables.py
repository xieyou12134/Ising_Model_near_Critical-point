from __future__ import annotations

import pytest
import torch

from ising_scale_diffusion.observables import (
    binder_cumulant,
    energy_density,
    open_axial_correlation,
)


def test_uniform_field_observables() -> None:
    spins = torch.ones((4, 4, 4))
    assert torch.all(energy_density(spins, periodic=True) == -2.0)
    assert torch.all(energy_density(spins, periodic=False) == -1.5)
    assert binder_cumulant(spins).item() == pytest.approx(2.0 / 3.0)
    torch.testing.assert_close(open_axial_correlation(spins, 3), torch.ones(4))


def test_global_spin_flip_preserves_energy_and_correlation() -> None:
    spins = torch.tensor([[[1, -1, 1], [-1, 1, -1], [1, -1, 1]]], dtype=torch.float32)
    torch.testing.assert_close(energy_density(spins), energy_density(-spins))
    torch.testing.assert_close(
        open_axial_correlation(spins, 2), open_axial_correlation(-spins, 2)
    )
