from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .config import load_config
from .io_utils import sha256_file


def verify_configs(
    config_paths: Iterable[str | Path], verify_checksums: bool = True
) -> dict[str, object]:
    configs = [load_config(path) for path in config_paths]
    if not configs:
        raise ValueError("At least one config is required")
    manifest_path = configs[0].manifest_dir / "parents.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    ids = [row["sample_id"] for row in manifest]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample_id in parents.csv")

    expected_total = 0
    observed_seeds: set[int] = set()
    verified_shards = 0
    for config in configs:
        expected_total += config.n_chains * config.n_samples_per_chain
        matching = [row for row in manifest if row["split"] == config.split]
        if len(matching) != config.n_chains * config.n_samples_per_chain:
            raise RuntimeError(
                f"Wrong parent count for {config.split}: {len(matching)}"
            )
        for row in matching:
            if int(row["parent_size"]) != config.parent_size:
                raise RuntimeError(f"Wrong parent_size in manifest: {row['sample_id']}")
            if row["config_sha256"] != config.source_sha256:
                raise RuntimeError(
                    f"Wrong config checksum in manifest: {row['sample_id']}"
                )
        for chain_id in range(config.n_chains):
            stem = config.split_dir / f"chain_{chain_id:03d}"
            shard = stem.with_suffix(".npy")
            meta_path = stem.with_suffix(".meta.json")
            metrics = stem.with_suffix(".metrics.npz")
            if not (shard.exists() and meta_path.exists() and metrics.exists()):
                raise RuntimeError(f"Incomplete chain: {stem}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            seed = int(meta["chain_seed"])
            if seed in observed_seeds:
                raise RuntimeError(f"Repeated chain seed: {seed}")
            observed_seeds.add(seed)
            array = np.load(shard, mmap_mode="r")
            expected_shape = (
                config.n_samples_per_chain,
                config.parent_size,
                config.parent_size,
            )
            if array.shape != expected_shape or array.dtype != np.int8:
                raise RuntimeError(f"Bad shard contract: {shard}")
            if verify_checksums and sha256_file(shard) != meta["sha256"]:
                raise RuntimeError(f"Bad checksum: {shard}")
            verified_shards += 1
    selected = [
        row for row in manifest if row["split"] in {config.split for config in configs}
    ]
    if len(selected) != expected_total:
        raise RuntimeError("Manifest total does not match configs")
    return {
        "passed": True,
        "parents": expected_total,
        "shards": verified_shards,
        "unique_chain_seeds": len(observed_seeds),
        "checksums_verified": verify_checksums,
    }


def verify_crop_manifest(
    path: str | Path, parent_manifest: str | Path
) -> dict[str, int | bool]:
    with Path(parent_manifest).open(newline="", encoding="utf-8") as handle:
        parents = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        crops = list(csv.DictReader(handle))
    crop_ids: set[str] = set()
    for crop in crops:
        if crop["crop_id"] in crop_ids:
            raise RuntimeError(f"Duplicate crop_id: {crop['crop_id']}")
        crop_ids.add(crop["crop_id"])
        parent = parents.get(crop["parent_id"])
        if parent is None or parent["split"] != crop["split"]:
            raise RuntimeError(f"Invalid parent reference: {crop['crop_id']}")
        if (
            crop["parent_sha256"] != parent["sha256"]
            or crop["shard_path"] != parent["shard_path"]
        ):
            raise RuntimeError(f"Stale parent metadata: {crop['crop_id']}")
        parent_size = int(parent["parent_size"])
        size, top, left = (
            int(crop["crop_size"]),
            int(crop["crop_top"]),
            int(crop["crop_left"]),
        )
        if (
            size <= 0
            or top < 0
            or left < 0
            or top + size > parent_size
            or left + size > parent_size
        ):
            raise RuntimeError(f"Crop crosses parent boundary: {crop['crop_id']}")
    return {"passed": True, "crops": len(crops), "unique_crops": len(crop_ids)}
