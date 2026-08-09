from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .rng import stable_i63, stable_u64


def apply_d4(field: np.ndarray, transform_id: int) -> np.ndarray:
    if not 0 <= transform_id < 8:
        raise ValueError("D4 transform_id must lie in [0, 8)")
    transformed = np.fliplr(field) if transform_id >= 4 else field
    return np.ascontiguousarray(np.rot90(transformed, transform_id % 4))


class ParentStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._arrays: dict[str, np.ndarray] = {}

    def get(self, shard_path: str) -> np.ndarray:
        if shard_path not in self._arrays:
            path = Path(shard_path)
            if not path.is_absolute():
                path = self.root / path
            self._arrays[shard_path] = np.load(path, mmap_mode="r")
        return self._arrays[shard_path]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ReplayableCropSource:
    """Counter-based online crops with one spatial width per microbatch."""

    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path,
        split: str,
        widths: tuple[int, ...],
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.store = ParentStore(root)
        self.split = split
        self.widths = tuple(widths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        rows = [row for row in _read_csv(self.manifest_path) if row["split"] == split]
        if not rows:
            raise ValueError(f"No rows for split={split!r} in {self.manifest_path}")
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["chain_id"])].append(row)
        self.chain_ids = tuple(sorted(grouped))
        self.rows_by_chain = {key: tuple(grouped[key]) for key in self.chain_ids}
        smallest_parent = min(int(row["parent_size"]) for row in rows)
        if max(self.widths) > smallest_parent:
            raise ValueError("A requested crop width exceeds the smallest parent field")

    def width_for_step(self, step: int) -> int:
        index = stable_u64(self.seed, "width", int(step)) % len(self.widths)
        return self.widths[index]

    def _sample(self, logical_index: int, width: int, namespace: str) -> dict[str, Any]:
        seed = stable_u64(self.seed, namespace, int(logical_index), int(width))
        generator = np.random.default_rng(seed)
        chain_id = self.chain_ids[int(generator.integers(len(self.chain_ids)))]
        candidates = self.rows_by_chain[chain_id]
        parent = candidates[int(generator.integers(len(candidates)))]
        parent_size = int(parent["parent_size"])
        maximum = parent_size - width
        top = int(generator.integers(maximum + 1))
        left = int(generator.integers(maximum + 1))
        transform_id = int(generator.integers(8))
        spin_flip = bool(generator.integers(2))

        parents = self.store.get(parent["shard_path"])
        field = np.asarray(
            parents[
                int(parent["index_in_chain"]), top : top + width, left : left + width
            ]
        )
        field = apply_d4(field, transform_id)
        if spin_flip:
            field = -field
        tokens = ((field.astype(np.int16) + 1) // 2).astype(np.int64)
        identity = (
            parent["sample_id"],
            top,
            left,
            width,
            transform_id,
            spin_flip,
            namespace,
            logical_index,
        )
        return {
            "clean_tokens": torch.from_numpy(np.ascontiguousarray(tokens)),
            "beta": float(parent["beta"]),
            "sample_id": stable_i63("sample", *identity),
            "noise_seed": stable_i63("noise", *identity),
            "source": {
                "parent_id": parent["sample_id"],
                "chain_id": chain_id,
                "crop_top": top,
                "crop_left": left,
                "crop_size": width,
                "transform_id": transform_id,
                "spin_flip": spin_flip,
            },
        }

    def batch(
        self,
        step: int,
        microbatch: int = 0,
        accumulation_steps: int = 1,
        width: int | None = None,
        namespace: str = "train",
    ) -> dict[str, Any]:
        selected_width = self.width_for_step(step) if width is None else int(width)
        global_microbatch = int(step) * int(accumulation_steps) + int(microbatch)
        first = (global_microbatch * self.world_size + self.rank) * self.batch_size
        samples = [
            self._sample(first + offset, selected_width, namespace)
            for offset in range(self.batch_size)
        ]
        return {
            "clean_tokens": torch.stack([sample["clean_tokens"] for sample in samples]),
            "beta": torch.tensor(
                [sample["beta"] for sample in samples], dtype=torch.float32
            ),
            "sample_ids": torch.tensor(
                [sample["sample_id"] for sample in samples], dtype=torch.int64
            ),
            "noise_seeds": torch.tensor(
                [sample["noise_seed"] for sample in samples], dtype=torch.int64
            ),
            "source_ids": [sample["source"] for sample in samples],
            "width": selected_width,
            "n_sites": self.batch_size * selected_width * selected_width,
        }


class FixedCropSource:
    """Frozen validation crops read from the Monte Carlo crop manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path,
        split: str,
        batch_size: int,
        seed: int,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.store = ParentStore(root)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in _read_csv(self.manifest_path):
            if row["split"] == split:
                grouped[int(row["crop_size"])].append(row)
        if not grouped:
            raise ValueError(
                f"No fixed crops for split={split!r} in {self.manifest_path}"
            )
        self.rows_by_width = {
            width: tuple(sorted(rows, key=lambda row: row["crop_id"]))
            for width, rows in grouped.items()
        }

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(sorted(self.rows_by_width))

    def batch(self, width: int, batch_index: int, t_index: int = 0) -> dict[str, Any]:
        rows = self.rows_by_width[int(width)]
        start = (int(batch_index) * self.batch_size) % len(rows)
        selected = [
            rows[(start + offset) % len(rows)] for offset in range(self.batch_size)
        ]
        tokens: list[torch.Tensor] = []
        for row in selected:
            parents = self.store.get(row["shard_path"])
            top = int(row["crop_top"])
            left = int(row["crop_left"])
            size = int(row["crop_size"])
            field = np.asarray(
                parents[
                    int(row["index_in_chain"]), top : top + size, left : left + size
                ]
            )
            clean = ((field.astype(np.int16) + 1) // 2).astype(np.int64)
            tokens.append(torch.from_numpy(np.ascontiguousarray(clean)))
        return {
            "clean_tokens": torch.stack(tokens),
            "beta": torch.tensor([float(row["beta"]) for row in selected]),
            "sample_ids": torch.tensor(
                [stable_i63("fixed", row["crop_id"]) for row in selected],
                dtype=torch.int64,
            ),
            "noise_seeds": torch.tensor(
                [
                    stable_i63("fixed-noise", self.seed, row["crop_id"], t_index)
                    for row in selected
                ],
                dtype=torch.int64,
            ),
            "source_ids": [{"crop_id": row["crop_id"]} for row in selected],
            "width": int(width),
            "n_sites": self.batch_size * int(width) * int(width),
        }
