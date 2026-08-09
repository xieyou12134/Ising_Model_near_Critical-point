from __future__ import annotations

import numpy as np

from critical_ising_mc.diagnostics import (
    effective_sample_size,
    integrated_autocorrelation_time,
    split_rhat,
)
from critical_ising_mc.parent_size import open_axial_correlation


def test_iat_and_ess_for_alternating_series() -> None:
    series = np.tile(np.asarray([-1.0, 1.0]), 128)
    tau = integrated_autocorrelation_time(series)
    _, ess = effective_sample_size(series)
    assert tau == 0.5
    assert ess == len(series)


def test_split_rhat_identical_distributions() -> None:
    generator = np.random.default_rng(7)
    chains = [generator.normal(size=1000) for _ in range(4)]
    assert split_rhat(chains) < 1.02


def test_open_correlation_does_not_wrap() -> None:
    spins = np.asarray([[1, 1, -1], [1, 1, -1], [1, 1, -1]], dtype=np.int8)
    correlation = open_axial_correlation(spins, max_distance=2)
    expected_horizontal_r1 = 0.0
    expected_vertical_r1 = 1.0
    assert correlation[0] == 1.0
    assert correlation[1] == 0.5 * (expected_horizontal_r1 + expected_vertical_r1)
