from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from ising_scale_diffusion.data import ReplayableCropSource


def _manifest(root: Path) -> Path:
    data = root / "data"
    data.mkdir()
    field = np.ones((3, 8, 8), dtype=np.int8)
    field[:, :4, :4] = -1
    np.save(data / "chain_000.npy", field)
    path = root / "parents.csv"
    fields = [
        "sample_id",
        "split",
        "chain_id",
        "index_in_chain",
        "beta",
        "parent_size",
        "shard_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(3):
            writer.writerow(
                {
                    "sample_id": f"train-c000-p{index:05d}",
                    "split": "train",
                    "chain_id": 0,
                    "index_in_chain": index,
                    "beta": 0.44068679350977147,
                    "parent_size": 8,
                    "shard_path": "data/chain_000.npy",
                }
            )
    return path


def test_replayable_batches_are_deterministic(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    source = ReplayableCropSource(
        manifest, tmp_path, "train", (4, 6), batch_size=2, seed=17
    )
    first = source.batch(step=5, microbatch=1, accumulation_steps=2)
    second = source.batch(step=5, microbatch=1, accumulation_steps=2)
    assert first["width"] == second["width"]
    assert torch.equal(first["clean_tokens"], second["clean_tokens"])
    assert torch.equal(first["sample_ids"], second["sample_ids"])
    assert torch.equal(first["noise_seeds"], second["noise_seeds"])
    assert set(torch.unique(first["clean_tokens"]).tolist()).issubset({0, 1})


def test_width_schedule_is_step_determined(tmp_path: Path) -> None:
    source = ReplayableCropSource(
        _manifest(tmp_path), tmp_path, "train", (4, 6), batch_size=1, seed=21
    )
    sequence = [source.width_for_step(step) for step in range(20)]
    assert sequence == [source.width_for_step(step) for step in range(20)]
    assert set(sequence) == {4, 6}
