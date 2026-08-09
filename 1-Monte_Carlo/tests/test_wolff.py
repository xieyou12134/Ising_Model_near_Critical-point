from __future__ import annotations

from collections import defaultdict

import numpy as np

from critical_ising_mc.exact import enumerate_ising
from critical_ising_mc.rng import initial_spins
from critical_ising_mc.wolff import (
    add_probability,
    energy_per_site,
    magnetization,
    make_workspace,
    wolff_flip,
)

BETA_C = 0.44068679350977147


def _trajectory(seed: int, steps: int = 100) -> tuple[np.ndarray, list[int]]:
    spins = initial_spins(8, "random", seed=1234)
    workspace = make_workspace(spins.size)
    rng_state = np.uint64(seed)
    sizes = []
    for _ in range(steps):
        rng_state, size, workspace.mark_token = wolff_flip(
            spins,
            add_probability(BETA_C),
            rng_state,
            workspace.stack,
            workspace.cluster,
            workspace.marks,
            workspace.mark_token,
        )
        sizes.append(size)
        assert 1 <= size <= spins.size
        assert np.all((spins == -1) | (spins == 1))
    return spins.copy(), sizes


def test_reproducible_and_seed_separated() -> None:
    spins_a, sizes_a = _trajectory(42)
    spins_b, sizes_b = _trajectory(42)
    spins_c, sizes_c = _trajectory(43)
    assert np.array_equal(spins_a, spins_b)
    assert sizes_a == sizes_b
    assert sizes_a != sizes_c or not np.array_equal(spins_a, spins_c)


def test_uniform_lattice_forms_one_cluster_at_probability_one() -> None:
    spins = np.ones((4, 4), dtype=np.int8)
    workspace = make_workspace(spins.size)
    _, size, workspace.mark_token = wolff_flip(
        spins,
        1.0,
        np.uint64(7),
        workspace.stack,
        workspace.cluster,
        workspace.marks,
        workspace.mark_token,
    )
    assert size == 16
    assert np.all(spins == -1)
    assert energy_per_site(spins) == -2.0
    assert magnetization(spins) == -1.0


def test_l4_matches_exact_energy_and_magnetization_distributions() -> None:
    exact = enumerate_ising(4, BETA_C)
    exact_energy: dict[int, float] = defaultdict(float)
    exact_magnetization: dict[int, float] = defaultdict(float)
    for energy, magnetization_value, probability in zip(
        exact["energy"], exact["magnetization"], exact["probability"], strict=True
    ):
        exact_energy[int(energy)] += float(probability)
        exact_magnetization[int(magnetization_value)] += float(probability)

    spins = initial_spins(4, "random", seed=99)
    workspace = make_workspace(spins.size)
    rng_state = np.uint64(1234567)
    p_add = add_probability(BETA_C)
    energy_counts: dict[int, int] = defaultdict(int)
    magnetization_counts: dict[int, int] = defaultdict(int)
    for step in range(30_500):
        rng_state, _, workspace.mark_token = wolff_flip(
            spins,
            p_add,
            rng_state,
            workspace.stack,
            workspace.cluster,
            workspace.marks,
            workspace.mark_token,
        )
        if step >= 500:
            energy_counts[round(energy_per_site(spins) * spins.size)] += 1
            magnetization_counts[round(magnetization(spins) * spins.size)] += 1

    n_samples = 30_000
    energy_tv = 0.5 * sum(
        abs(energy_counts.get(value, 0) / n_samples - probability)
        for value, probability in exact_energy.items()
    )
    magnetization_tv = 0.5 * sum(
        abs(magnetization_counts.get(value, 0) / n_samples - probability)
        for value, probability in exact_magnetization.items()
    )
    assert energy_tv < 0.035
    assert magnetization_tv < 0.035
