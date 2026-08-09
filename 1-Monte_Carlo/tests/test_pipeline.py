from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from critical_ising_mc.config import load_config
from critical_ising_mc.crops import (
    FixedCropDataset,
    OnlineTrainingCropDataset,
    apply_d4,
    create_fixed_crop_manifests,
)
from critical_ising_mc.diagnostics import diagnose_configs, diagnose_split
from critical_ising_mc.sampling import generate_split
from critical_ising_mc.verification import verify_configs


def _write_config(
    root: Path, split: str, split_id: int, chains: int = 2, samples: int = 48
) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{split}.yaml"
    path.write_text(
        "\n".join(
            [
                f"run_name: test_{split}",
                f"split: {split}",
                f"split_id: {split_id}",
                "beta: 0.44068679350977147",
                "parent_size: 4",
                f"n_chains: {chains}",
                f"n_samples_per_chain: {samples}",
                "adaptation_sweeps: 2.0",
                "pilot_cluster_steps: 8",
                "pilot_rounds: 2",
                "burnin_sweeps: 10.0",
                "sweeps_between_samples: 2.0",
                f"base_seed: {1000 + split_id}",
                "dtype: int8",
                "output_root: ../data/test_L4",
                "manifest_dir: ../manifests",
                "report_dir: ../reports",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_generate_resume_manifest_verify_and_invariants(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "train", 0)
    first = generate_split(config_path, workers=1)
    second = generate_split(config_path, workers=1)
    assert {row["status"] for row in first} == {"generated"}
    assert {row["status"] for row in second} == {"reused"}

    config = load_config(config_path)
    manifest_path = config.manifest_dir / "parents.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 96
    assert len({row["sample_id"] for row in rows}) == 96
    assert verify_configs([config_path])["passed"]
    with np.load(config.split_dir / "chain_000.metrics.npz") as metrics:
        assert metrics["gap_cluster_sizes"].shape[0] == 48
        assert np.array_equal(
            np.sum(metrics["gap_cluster_sizes"], axis=1), metrics["realized_gap_sites"]
        )
        assert np.all(np.diff(metrics["cumulative_updated_sites"]) > 0)

    _, summary, _ = diagnose_split(config)
    assert summary["invariants"]["g0_error"] < 1e-10
    assert summary["invariants"]["parseval_error"] < 1e-10
    assert summary["invariants"]["energy_correlation_error"] < 1e-10
    assert summary["invariants"]["s0_error"] < 1e-10


def test_online_and_fixed_crop_datasets(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "train", 0, chains=2, samples=8)
    generate_split(config_path, workers=1)
    config = load_config(config_path)
    online = OnlineTrainingCropDataset(
        config.manifest_dir / "parents.csv",
        config.monte_carlo_root,
        sizes=(2, 3),
        epoch_size=20,
        base_seed=22,
    )
    sample_a = online[3]
    sample_b = online[3]
    assert np.array_equal(sample_a["spin"], sample_b["spin"])
    assert sample_a["spin"].shape[0] in (2, 3)
    assert 0 <= sample_a["transform_id"] < 16

    row = next(
        csv.DictReader((config.manifest_dir / "parents.csv").open(encoding="utf-8"))
    )
    fixed_path = config.manifest_dir / "fixed.csv"
    with fixed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "crop_id": "fixed-0",
                "split": "train",
                "chain_id": row["chain_id"],
                "parent_id": row["sample_id"],
                "index_in_chain": row["index_in_chain"],
                "crop_size": 2,
                "crop_top": 1,
                "crop_left": 1,
                "transform_id": 0,
                "shard_path": row["shard_path"],
                "parent_sha256": row["sha256"],
                "beta": row["beta"],
            }
        )
    fixed = FixedCropDataset(fixed_path, config.monte_carlo_root)
    assert len(fixed) == 1
    assert fixed[0]["spin"].shape == (2, 2)


def test_d4_preserves_values_and_shape() -> None:
    source = np.arange(9).reshape(3, 3)
    variants = [apply_d4(source, transform_id) for transform_id in range(8)]
    assert all(array.shape == (3, 3) for array in variants)
    assert all(
        np.array_equal(np.sort(array.ravel()), np.arange(9)) for array in variants
    )


def test_all_split_diagnostics_and_frozen_crop_manifests(tmp_path: Path) -> None:
    configs = [
        _write_config(tmp_path, split, split_id, chains=2, samples=32)
        for split_id, split in enumerate(("train", "val", "reference_a", "reference_b"))
    ]
    for config in configs:
        generate_split(config, workers=1)
    result = diagnose_configs(configs)
    assert set(result["splits"]) == {"train", "val", "reference_a", "reference_b"}
    report_dir = load_config(configs[0]).report_dir
    assert (report_dir / "chain_diagnostics.csv").exists()
    assert (report_dir / "observables.npz").exists()
    assert (report_dir / "validation.md").exists()

    spec = tmp_path / "configs" / "crops.yaml"
    spec.write_text(
        """base_seed: 404
training_sizes: [2, 3]
validation_sizes: [2, 3]
reference_sizes: [2, 3]
validation_crops_per_parent: 1
reference_crops_per_parent: 1
""",
        encoding="utf-8",
    )
    val_path, reference_path = create_fixed_crop_manifests(configs, spec)
    assert len(FixedCropDataset(val_path, tmp_path)) == 2 * 32 * 2
    assert len(FixedCropDataset(reference_path, tmp_path)) == 2 * 2 * 32 * 2
