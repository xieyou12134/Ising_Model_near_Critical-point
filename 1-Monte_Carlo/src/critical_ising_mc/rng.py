from __future__ import annotations

import numpy as np


def chain_seeds(base_seed: int, split_id: int, chain_id: int) -> tuple[int, int, int]:
    """Derive independent recorded, initialization, and Wolff seeds."""
    sequence = np.random.SeedSequence([base_seed, split_id, chain_id])
    chain_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
    init_sequence, wolff_sequence = sequence.spawn(2)
    init_seed = int(init_sequence.generate_state(1, dtype=np.uint64)[0])
    wolff_seed = int(wolff_sequence.generate_state(1, dtype=np.uint64)[0])
    if wolff_seed == 0:
        wolff_seed = 0x9E3779B97F4A7C15
    return chain_seed, init_seed, wolff_seed


def initial_spins(size: int, initial_state: str, seed: int) -> np.ndarray:
    if initial_state == "ordered_plus":
        return np.ones((size, size), dtype=np.int8)
    if initial_state == "random":
        generator = np.random.default_rng(seed)
        return np.where(
            generator.integers(0, 2, size=(size, size), dtype=np.int8), 1, -1
        ).astype(np.int8)
    raise ValueError(f"Unknown initial state: {initial_state}")
