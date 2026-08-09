from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig, load_config
from .io_utils import (
    atomic_savez,
    atomic_write_csv,
    atomic_write_json,
    freeze_config,
    portable_path,
    provenance,
    sha256_file,
)
from .rng import chain_seeds, initial_spins
from .wolff import (
    add_probability,
    energy_per_site,
    magnetization,
    make_workspace,
    run_n_flips,
    run_until_sites,
)

PARENT_FIELDS = [
    "sample_id",
    "split",
    "chain_id",
    "index_in_chain",
    "beta",
    "parent_size",
    "seed",
    "initial_state",
    "gap_cluster_flips",
    "realized_gap_sweeps",
    "energy",
    "magnetization",
    "shard_path",
    "sha256",
    "config_sha256",
]


def _chain_paths(config: RunConfig, chain_id: int) -> tuple[Path, Path, Path]:
    stem = config.split_dir / f"chain_{chain_id:03d}"
    return (
        stem.with_suffix(".npy"),
        stem.with_suffix(".metrics.npz"),
        stem.with_suffix(".meta.json"),
    )


def _valid_completed_chain(config: RunConfig, chain_id: int) -> dict[str, Any] | None:
    shard_path, metrics_path, meta_path = _chain_paths(config, chain_id)
    if not (shard_path.exists() and metrics_path.exists() and meta_path.exists()):
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("config_sha256") != config.source_sha256:
        return None
    expected_shape = [
        config.n_samples_per_chain,
        config.parent_size,
        config.parent_size,
    ]
    if metadata.get("shape") != expected_shape or metadata.get("sha256") != sha256_file(
        shard_path
    ):
        return None
    array = np.load(shard_path, mmap_mode="r")
    if list(array.shape) != expected_shape or array.dtype != np.int8:
        return None
    return metadata


def sample_chain(
    config_path: str, chain_id: int, force: bool = False
) -> dict[str, Any]:
    config = load_config(config_path)
    if not 0 <= chain_id < config.n_chains:
        raise ValueError(f"chain_id {chain_id} outside [0, {config.n_chains})")
    config.split_dir.mkdir(parents=True, exist_ok=True)
    shard_path, metrics_path, meta_path = _chain_paths(config, chain_id)

    if not force:
        completed = _valid_completed_chain(config, chain_id)
        if completed is not None:
            return {**completed, "status": "reused"}

    for partial in config.split_dir.glob(f"chain_{chain_id:03d}*.partial"):
        partial.unlink()

    started = time.time()
    chain_seed, init_seed, wolff_seed = chain_seeds(
        config.base_seed, config.split_id, chain_id
    )
    initial_state = "random" if chain_id % 2 == 0 else "ordered_plus"
    spins = initial_spins(config.parent_size, initial_state, init_seed)
    workspace = make_workspace(config.n_sites)
    rng_state = np.uint64(wolff_seed)
    p_add = add_probability(config.beta)

    adaptation_target = math.ceil(config.adaptation_sweeps * config.n_sites)
    rng_state, adaptation_sites, adaptation_flips = run_until_sites(
        spins, p_add, rng_state, workspace, adaptation_target
    )

    rng_state, pre_pilot_sites, pre_pilot_sizes = run_n_flips(
        spins, p_add, rng_state, workspace, config.pilot_flips
    )
    pre_pilot_mean = float(np.mean(pre_pilot_sizes))
    pre_pilot_round_means = np.mean(
        pre_pilot_sizes.reshape(config.pilot_rounds, config.pilot_cluster_steps), axis=1
    )
    burnin_flips = math.ceil(config.burnin_sweeps * config.n_sites / pre_pilot_mean)
    rng_state, burnin_sites, _ = run_n_flips(
        spins, p_add, rng_state, workspace, burnin_flips
    )

    rng_state, post_pilot_sites, post_pilot_sizes = run_n_flips(
        spins, p_add, rng_state, workspace, config.pilot_flips
    )
    post_pilot_mean = float(np.mean(post_pilot_sizes))
    post_pilot_round_means = np.mean(
        post_pilot_sizes.reshape(config.pilot_rounds, config.pilot_cluster_steps),
        axis=1,
    )
    gap_flips = math.ceil(
        config.sweeps_between_samples * config.n_sites / post_pilot_mean
    )

    temporary_shard = shard_path.with_name(shard_path.name + ".partial")
    output = np.lib.format.open_memmap(
        temporary_shard,
        mode="w+",
        dtype=np.int8,
        shape=(config.n_samples_per_chain, config.parent_size, config.parent_size),
    )
    energies = np.empty(config.n_samples_per_chain, dtype=np.float64)
    magnetizations = np.empty(config.n_samples_per_chain, dtype=np.float64)
    realized_gap_sites = np.empty(config.n_samples_per_chain, dtype=np.int64)
    gap_cluster_sizes = np.empty(
        (config.n_samples_per_chain, gap_flips), dtype=np.int32
    )

    try:
        for sample_index in range(config.n_samples_per_chain):
            rng_state, changed_sites, cluster_sizes = run_n_flips(
                spins, p_add, rng_state, workspace, gap_flips
            )
            output[sample_index] = spins
            energies[sample_index] = energy_per_site(spins)
            magnetizations[sample_index] = magnetization(spins)
            realized_gap_sites[sample_index] = changed_sites
            gap_cluster_sizes[sample_index] = cluster_sizes
        output.flush()
    except BaseException:
        del output
        temporary_shard.unlink(missing_ok=True)
        raise
    del output
    try:
        with temporary_shard.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_shard, shard_path)
    except BaseException:
        temporary_shard.unlink(missing_ok=True)
        raise

    sites_before_production = (
        adaptation_sites + pre_pilot_sites + burnin_sites + post_pilot_sites
    )
    cumulative_updated_sites = sites_before_production + np.cumsum(
        realized_gap_sites, dtype=np.int64
    )
    atomic_savez(
        metrics_path,
        energy=energies,
        magnetization=magnetizations,
        realized_gap_sites=realized_gap_sites,
        cumulative_updated_sites=cumulative_updated_sites,
        gap_cluster_sizes=gap_cluster_sizes,
        pre_pilot_cluster_sizes=pre_pilot_sizes,
        post_pilot_cluster_sizes=post_pilot_sizes,
    )
    shard_sha = sha256_file(shard_path)
    metadata = {
        "run_name": config.run_name,
        "split": config.split,
        "chain_id": chain_id,
        "chain_seed": chain_seed,
        "init_seed": init_seed,
        "wolff_seed": wolff_seed,
        "final_rng_state": int(rng_state),
        "initial_state": initial_state,
        "shape": [config.n_samples_per_chain, config.parent_size, config.parent_size],
        "dtype": "int8",
        "beta": config.beta,
        "config_sha256": config.source_sha256,
        "sha256": shard_sha,
        "shard_path": portable_path(shard_path, config.monte_carlo_root),
        "metrics_path": portable_path(metrics_path, config.monte_carlo_root),
        "adaptation_flips": adaptation_flips,
        "adaptation_realized_sweeps": adaptation_sites / config.n_sites,
        "pre_pilot_flips": config.pilot_flips,
        "pre_pilot_realized_sweeps": pre_pilot_sites / config.n_sites,
        "pre_pilot_mean_cluster_size": pre_pilot_mean,
        "pre_pilot_round_mean_cluster_sizes": pre_pilot_round_means.tolist(),
        "burnin_flips": burnin_flips,
        "burnin_realized_sweeps": burnin_sites / config.n_sites,
        "post_pilot_flips": config.pilot_flips,
        "post_pilot_realized_sweeps": post_pilot_sites / config.n_sites,
        "post_pilot_mean_cluster_size": post_pilot_mean,
        "post_pilot_round_mean_cluster_sizes": post_pilot_round_means.tolist(),
        "gap_cluster_flips": gap_flips,
        "mean_realized_gap_sweeps": float(np.mean(realized_gap_sites) / config.n_sites),
        "elapsed_seconds": time.time() - started,
        "provenance": provenance(config.monte_carlo_root.parent),
    }
    atomic_write_json(meta_path, metadata)
    return {**metadata, "status": "generated"}


def generate_split(
    config_path: str | Path, workers: int = 1, force: bool = False
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    config.split_dir.mkdir(parents=True, exist_ok=True)
    config.manifest_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    freeze_config(
        config.source_path,
        config.split_dir / "run_config.yaml",
        config.source_sha256,
    )
    atomic_write_json(
        config.split_dir / "run_provenance.json",
        {
            "config": config.metadata(),
            "environment": provenance(config.monte_carlo_root.parent),
        },
    )

    if workers <= 1:
        results = [
            sample_chain(str(config.source_path), chain_id, force)
            for chain_id in range(config.n_chains)
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(workers, config.n_chains)) as executor:
            futures = {
                executor.submit(
                    sample_chain, str(config.source_path), chain_id, force
                ): chain_id
                for chain_id in range(config.n_chains)
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: int(item["chain_id"]))

    rebuild_parent_manifest(config)
    return results


def rebuild_parent_manifest(config: RunConfig) -> Path:
    rows: list[dict[str, Any]] = []
    canonical_manifest_dir = (config.monte_carlo_root / "manifests").resolve()
    if config.manifest_dir == canonical_manifest_dir:
        meta_paths = sorted(
            (config.monte_carlo_root / "data").glob("*/*/chain_*.meta.json")
        )
    else:
        meta_paths = sorted(config.output_root.glob("*/chain_*.meta.json"))
    for meta_path in meta_paths:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            config.manifest_dir == canonical_manifest_dir
            and metadata.get("split") == "smoke"
        ):
            continue
        shard_path = config.monte_carlo_root / metadata["shard_path"]
        metrics_path = config.monte_carlo_root / metadata["metrics_path"]
        if not shard_path.exists() or not metrics_path.exists():
            continue
        with np.load(metrics_path) as metrics:
            energies = metrics["energy"]
            magnetizations = metrics["magnetization"]
            realized = metrics["realized_gap_sites"]
        parent_size = int(metadata["shape"][1])
        n_sites = parent_size * parent_size
        for index in range(int(metadata["shape"][0])):
            rows.append(
                {
                    "sample_id": f"{metadata['split']}-c{int(metadata['chain_id']):03d}-p{index:05d}",
                    "split": metadata["split"],
                    "chain_id": metadata["chain_id"],
                    "index_in_chain": index,
                    "beta": metadata["beta"],
                    "parent_size": parent_size,
                    "seed": metadata["chain_seed"],
                    "initial_state": metadata["initial_state"],
                    "gap_cluster_flips": metadata["gap_cluster_flips"],
                    "realized_gap_sweeps": float(realized[index]) / n_sites,
                    "energy": float(energies[index]),
                    "magnetization": float(magnetizations[index]),
                    "shard_path": metadata["shard_path"],
                    "sha256": metadata["sha256"],
                    "config_sha256": metadata["config_sha256"],
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["split"]),
            int(row["chain_id"]),
            int(row["index_in_chain"]),
        )
    )
    destination = config.manifest_dir / "parents.csv"
    atomic_write_csv(destination, PARENT_FIELDS, rows)
    return destination


def read_parent_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
