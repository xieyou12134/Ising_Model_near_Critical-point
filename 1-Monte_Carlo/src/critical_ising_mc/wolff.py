from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - exercised by installation checks
    raise ImportError(
        "critical-ising-mc requires numba; install with `pip install -e .`"
    ) from exc


@dataclass
class WolffWorkspace:
    stack: np.ndarray
    cluster: np.ndarray
    marks: np.ndarray
    mark_token: int = 0


def make_workspace(n_sites: int) -> WolffWorkspace:
    return WolffWorkspace(
        stack=np.empty(n_sites, dtype=np.int32),
        cluster=np.empty(n_sites, dtype=np.int32),
        marks=np.zeros(n_sites, dtype=np.int32),
    )


@njit(cache=True)
def _next_u64(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    value = state * np.uint64(2685821657736338717)
    return state, value


@njit(cache=True)
def _uniform01(state: np.uint64) -> tuple[np.uint64, float]:
    state, value = _next_u64(state)
    # Converting a uint64 above 2**63 directly to float is backend-dependent in
    # some Numba releases. Masking to 53 bits keeps the conversion unambiguous.
    mantissa = value & np.uint64(9007199254740991)
    return state, float(mantissa) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _wolff_flip_kernel(
    spins: np.ndarray,
    p_add: float,
    rng_state: np.uint64,
    stack: np.ndarray,
    cluster: np.ndarray,
    marks: np.ndarray,
    mark_token: int,
) -> tuple[np.uint64, int, int]:
    size_l = spins.shape[0]
    n_sites = size_l * size_l
    flat = spins.reshape(n_sites)

    mark_token += 1
    if mark_token >= 2_147_483_647:
        marks[:] = 0
        mark_token = 1

    rng_state, random_value = _next_u64(rng_state)
    seed = int(random_value % np.uint64(n_sites))
    target_spin = flat[seed]
    stack[0] = seed
    marks[seed] = mark_token
    stack_size = 1
    cluster_size = 0

    while stack_size:
        stack_size -= 1
        site = stack[stack_size]
        cluster[cluster_size] = site
        cluster_size += 1
        row = site // size_l
        col = site - row * size_l

        for direction in range(4):
            if direction == 0:
                neighbor = row * size_l + ((col + 1) % size_l)
            elif direction == 1:
                neighbor = row * size_l + ((col - 1 + size_l) % size_l)
            elif direction == 2:
                neighbor = ((row + 1) % size_l) * size_l + col
            else:
                neighbor = ((row - 1 + size_l) % size_l) * size_l + col

            if marks[neighbor] != mark_token and flat[neighbor] == target_spin:
                rng_state, uniform = _uniform01(rng_state)
                if uniform < p_add:
                    marks[neighbor] = mark_token
                    stack[stack_size] = neighbor
                    stack_size += 1

    for index in range(cluster_size):
        flat[cluster[index]] = -flat[cluster[index]]
    return rng_state, cluster_size, mark_token


def wolff_flip(
    spins: np.ndarray,
    p_add: float,
    rng_state: np.uint64 | int,
    stack: np.ndarray,
    cluster: np.ndarray,
    marks: np.ndarray,
    mark_token: int,
) -> tuple[np.uint64, int, int]:
    """Run one flip while preserving uint64 state across the Python/JIT boundary."""
    state, size, token = _wolff_flip_kernel(
        spins, p_add, np.uint64(rng_state), stack, cluster, marks, mark_token
    )
    return np.uint64(state), int(size), int(token)


@njit(cache=True)
def energy_per_site(spins: np.ndarray) -> float:
    size_l = spins.shape[0]
    total = 0
    for row in range(size_l):
        for col in range(size_l):
            total -= int(spins[row, col]) * int(spins[row, (col + 1) % size_l])
            total -= int(spins[row, col]) * int(spins[(row + 1) % size_l, col])
    return total / float(size_l * size_l)


@njit(cache=True)
def magnetization(spins: np.ndarray) -> float:
    return float(np.sum(spins)) / float(spins.size)


def add_probability(beta: float) -> float:
    return 1.0 - math.exp(-2.0 * beta)


def run_n_flips(
    spins: np.ndarray,
    p_add: float,
    rng_state: np.uint64,
    workspace: WolffWorkspace,
    n_flips: int,
) -> tuple[np.uint64, int, np.ndarray]:
    sizes = np.empty(n_flips, dtype=np.int32)
    token = workspace.mark_token
    for index in range(n_flips):
        rng_state, cluster_size, token = wolff_flip(
            spins,
            p_add,
            rng_state,
            workspace.stack,
            workspace.cluster,
            workspace.marks,
            token,
        )
        sizes[index] = cluster_size
    workspace.mark_token = token
    return rng_state, int(np.sum(sizes, dtype=np.int64)), sizes


def run_until_sites(
    spins: np.ndarray,
    p_add: float,
    rng_state: np.uint64,
    workspace: WolffWorkspace,
    target_sites: int,
) -> tuple[np.uint64, int, int]:
    total_sites = 0
    flips = 0
    token = workspace.mark_token
    while total_sites < target_sites:
        rng_state, cluster_size, token = wolff_flip(
            spins,
            p_add,
            rng_state,
            workspace.stack,
            workspace.cluster,
            workspace.marks,
            token,
        )
        total_sites += cluster_size
        flips += 1
    workspace.mark_token = token
    return rng_state, total_sites, flips
