from __future__ import annotations

import numpy as np


def enumerate_ising(size_l: int, beta: float) -> dict[str, np.ndarray | float]:
    """Exact finite periodic Ising distribution for tiny lattices (tests only)."""
    n_sites = size_l * size_l
    if n_sites > 20:
        raise ValueError("Exact enumeration is limited to at most 20 spins")
    state_ids = np.arange(1 << n_sites, dtype=np.uint64)
    bit_positions = np.arange(n_sites, dtype=np.uint64)
    spins = (((state_ids[:, None] >> bit_positions[None, :]) & 1) * 2 - 1).astype(
        np.int8
    )
    spins = spins.reshape(-1, size_l, size_l)
    energies = -np.sum(
        spins * np.roll(spins, -1, axis=1) + spins * np.roll(spins, -1, axis=2),
        axis=(1, 2),
    )
    magnetizations = np.sum(spins, axis=(1, 2))
    log_weights = -beta * energies.astype(np.float64)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    probabilities = weights / np.sum(weights)
    return {
        "energy": energies,
        "magnetization": magnetizations,
        "probability": probabilities,
        "mean_energy_per_site": float(np.dot(probabilities, energies) / n_sites),
        "mean_m2": float(np.dot(probabilities, (magnetizations / n_sites) ** 2)),
    }
