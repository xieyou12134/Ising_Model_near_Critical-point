from __future__ import annotations

import numpy as np

from critical_ising_mc.diagnostics import (
    effective_sample_size,
    integrated_autocorrelation_time,
    split_rhat,
)
from critical_ising_mc.fixed_background import correlation_permutation_pvalue
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


def test_chain_permutation_check_detects_a_split_shift() -> None:
    shared = np.asarray(
        [[0.8, 0.6, 0.4], [0.79, 0.61, 0.39], [0.81, 0.59, 0.41], [0.8, 0.6, 0.4]]
    )
    _, same_p, same_permutations = correlation_permutation_pvalue(shared, shared)
    _, shifted_p, shifted_permutations = correlation_permutation_pvalue(
        shared, shared - 0.25
    )
    assert same_p == 1.0
    assert same_permutations == 70
    assert shifted_p < 0.05
    assert shifted_permutations == 70
