from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import load_config
from .io_utils import atomic_write_csv
from .sampling import read_parent_manifest

CROP_FIELDS = [
    "crop_id",
    "split",
    "chain_id",
    "parent_id",
    "index_in_chain",
    "crop_size",
    "crop_top",
    "crop_left",
    "transform_id",
    "shard_path",
    "parent_sha256",
    "beta",
]


def _crop_seed(
    base_seed: int,
    role_id: int,
    size: int,
    chain_id: int,
    parent_index: int,
    repeat: int,
) -> int:
    sequence = np.random.SeedSequence(
        [base_seed, role_id, size, chain_id, parent_index, repeat]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _fixed_rows(
    parents: Iterable[dict[str, str]],
    sizes: list[int],
    repeats: int,
    base_seed: int,
    role_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent in parents:
        parent_size = int(parent["parent_size"])
        for size in sizes:
            if size > parent_size:
                raise ValueError(f"Crop size {size} exceeds parent size {parent_size}")
            for repeat in range(repeats):
                seed = _crop_seed(
                    base_seed,
                    role_id,
                    size,
                    int(parent["chain_id"]),
                    int(parent["index_in_chain"]),
                    repeat,
                )
                generator = np.random.default_rng(seed)
                maximum = parent_size - size
                top = int(generator.integers(0, maximum + 1))
                left = int(generator.integers(0, maximum + 1))
                crop_id = f"{parent['sample_id']}-w{size:03d}-r{repeat:02d}"
                rows.append(
                    {
                        "crop_id": crop_id,
                        "split": parent["split"],
                        "chain_id": parent["chain_id"],
                        "parent_id": parent["sample_id"],
                        "index_in_chain": parent["index_in_chain"],
                        "crop_size": size,
                        "crop_top": top,
                        "crop_left": left,
                        "transform_id": 0,
                        "shard_path": parent["shard_path"],
                        "parent_sha256": parent["sha256"],
                        "beta": parent["beta"],
                    }
                )
    return rows


def create_fixed_crop_manifests(
    config_paths: Iterable[str | Path], crop_spec_path: str | Path
) -> tuple[Path, Path]:
    configs = {
        config.split: config for config in (load_config(path) for path in config_paths)
    }
    required = {"val", "reference_a", "reference_b"}
    missing = sorted(required.difference(configs))
    if missing:
        raise ValueError(f"Missing configs for crop manifests: {', '.join(missing)}")
    manifest_dirs = {config.manifest_dir for config in configs.values()}
    if len(manifest_dirs) != 1:
        raise ValueError("All configs must use one manifest_dir")
    manifest_dir = manifest_dirs.pop()
    parents_path = manifest_dir / "parents.csv"
    parents = read_parent_manifest(parents_path)

    spec_path = Path(crop_spec_path).resolve()
    spec = yaml.safe_load(spec_path.read_bytes())
    val_sizes = [int(value) for value in spec["validation_sizes"]]
    reference_sizes = [int(value) for value in spec["reference_sizes"]]
    val_repeats = int(spec.get("validation_crops_per_parent", 1))
    reference_repeats = int(spec.get("reference_crops_per_parent", 1))
    base_seed = int(spec["base_seed"])

    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for parent in parents:
        by_split[parent["split"]].append(parent)
    val_rows = _fixed_rows(
        by_split["val"], val_sizes, val_repeats, base_seed, role_id=1
    )
    reference_rows = _fixed_rows(
        by_split["reference_a"],
        reference_sizes,
        reference_repeats,
        base_seed,
        role_id=2,
    ) + _fixed_rows(
        by_split["reference_b"],
        reference_sizes,
        reference_repeats,
        base_seed,
        role_id=3,
    )

    val_path = manifest_dir / "val_crops.csv"
    reference_path = manifest_dir / "reference_crops.csv"
    atomic_write_csv(val_path, CROP_FIELDS, val_rows)
    atomic_write_csv(reference_path, CROP_FIELDS, reference_rows)
    return val_path, reference_path


def apply_d4(spins: np.ndarray, transform_id: int) -> np.ndarray:
    if not 0 <= transform_id < 8:
        raise ValueError("D4 transform_id must be in [0, 8)")
    result = spins
    if transform_id >= 4:
        result = np.fliplr(result)
    result = np.rot90(result, transform_id % 4)
    return np.ascontiguousarray(result)


class _ShardCache:
    def __init__(self, monte_carlo_root: str | Path):
        self.root = Path(monte_carlo_root).resolve()
        self._arrays: dict[str, np.ndarray] = {}

    def get(self, shard_path: str) -> np.ndarray:
        if shard_path not in self._arrays:
            path = Path(shard_path)
            if not path.is_absolute():
                path = self.root / path
            self._arrays[shard_path] = np.load(path, mmap_mode="r")
        return self._arrays[shard_path]


class FixedCropDataset:
    """Dataset backed by a frozen val/reference crop CSV.

    It implements the standard ``__len__``/``__getitem__`` protocol and can be
    passed directly to a PyTorch DataLoader; PyTorch is intentionally optional.
    """

    def __init__(self, manifest_path: str | Path, monte_carlo_root: str | Path):
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.cache = _ShardCache(monte_carlo_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        parent = self.cache.get(row["shard_path"])[int(row["index_in_chain"])]
        top, left, size = (
            int(row["crop_top"]),
            int(row["crop_left"]),
            int(row["crop_size"]),
        )
        spin = np.array(
            parent[top : top + size, left : left + size], dtype=np.int8, copy=True
        )
        transform_id = int(row["transform_id"])
        if transform_id:
            spin = apply_d4(spin, transform_id)
        return {
            "spin": spin,
            "beta": np.float64(row["beta"]),
            "split": row["split"],
            "chain_id": int(row["chain_id"]),
            "parent_id": row["parent_id"],
            "crop_top": top,
            "crop_left": left,
            "transform_id": transform_id,
        }


class OnlineTrainingCropDataset:
    """Deterministic-by-index online training crops with chain-balanced sampling."""

    def __init__(
        self,
        parent_manifest: str | Path,
        monte_carlo_root: str | Path,
        sizes: Iterable[int] = (32, 48, 64),
        epoch_size: int = 100_000,
        base_seed: int = 2026080950,
    ):
        parents = [
            row
            for row in read_parent_manifest(parent_manifest)
            if row["split"] == "train"
        ]
        if not parents:
            raise ValueError("No train parents found")
        self.by_chain: dict[int, list[dict[str, str]]] = defaultdict(list)
        for parent in parents:
            self.by_chain[int(parent["chain_id"])].append(parent)
        self.chain_ids = sorted(self.by_chain)
        self.sizes = tuple(int(value) for value in sizes)
        self.epoch_size = int(epoch_size)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.cache = _ShardCache(monte_carlo_root)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence = np.random.SeedSequence([self.base_seed, self.epoch, int(index)])
        generator = np.random.default_rng(sequence)
        size = self.sizes[int(generator.integers(len(self.sizes)))]
        chain_id = self.chain_ids[int(generator.integers(len(self.chain_ids)))]
        chain_parents = self.by_chain[chain_id]
        parent_row = chain_parents[int(generator.integers(len(chain_parents)))]
        parent = self.cache.get(parent_row["shard_path"])[
            int(parent_row["index_in_chain"])
        ]
        maximum = parent.shape[0] - size
        top = int(generator.integers(0, maximum + 1))
        left = int(generator.integers(0, maximum + 1))
        d4_id = int(generator.integers(0, 8))
        spin_flip = int(generator.integers(0, 2))
        spin = apply_d4(parent[top : top + size, left : left + size], d4_id)
        if spin_flip:
            spin = -spin
        transform_id = d4_id + 8 * spin_flip
        return {
            "spin": np.asarray(spin, dtype=np.int8),
            "beta": np.float64(parent_row["beta"]),
            "split": "train",
            "chain_id": chain_id,
            "parent_id": parent_row["sample_id"],
            "crop_top": top,
            "crop_left": left,
            "transform_id": transform_id,
        }
