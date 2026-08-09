from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .crops import _crop_seed
from .io_utils import atomic_savez
from .sampling import read_parent_manifest


def open_axial_correlation(crop: np.ndarray, max_distance: int = 64) -> np.ndarray:
    max_distance = min(max_distance, crop.shape[0] - 1, crop.shape[1] - 1)
    values = np.empty(max_distance + 1, dtype=np.float64)
    values[0] = 1.0
    for distance in range(1, max_distance + 1):
        horizontal = np.mean(crop[:, :-distance] * crop[:, distance:])
        vertical = np.mean(crop[:-distance, :] * crop[distance:, :])
        values[distance] = 0.5 * float(horizontal + vertical)
    return values


def _load_crop(
    root: Path, row: dict[str, str], cache: dict[str, np.ndarray]
) -> np.ndarray:
    shard_key = row["shard_path"]
    if shard_key not in cache:
        shard_path = Path(shard_key)
        if not shard_path.is_absolute():
            shard_path = root / shard_path
        cache[shard_key] = np.load(shard_path, mmap_mode="r")
    parents = cache[shard_key]
    top, left, size = int(row["crop_top"]), int(row["crop_left"]), int(row["crop_size"])
    return np.asarray(
        parents[int(row["index_in_chain"]), top : top + size, left : left + size]
    )


def _mean_by_chain(
    root: Path, rows: list[dict[str, str]], max_distance: int
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    cache: dict[str, np.ndarray] = {}
    for row in rows:
        grouped[int(row["chain_id"])].append(
            open_axial_correlation(_load_crop(root, row, cache), max_distance)
        )
    chain_curves = np.stack(
        [np.mean(grouped[chain_id], axis=0) for chain_id in sorted(grouped)]
    )
    return np.mean(chain_curves, axis=0), chain_curves


def run_parent_size_check(
    config_dir: str | Path,
    crop_spec_path: str | Path | None = None,
    max_distance: int = 64,
) -> dict[str, Any]:
    config_directory = Path(config_dir).resolve()
    ref_a = load_config(config_directory / "critical_L512_reference_a.yaml")
    check = load_config(config_directory / "critical_L1024_parent_size_check.yaml")
    root = ref_a.monte_carlo_root
    if check.monte_carlo_root != root or check.manifest_dir != ref_a.manifest_dir:
        raise ValueError(
            "L=512 and L=1024 configs must share one Monte Carlo root and manifest"
        )
    manifest_path = ref_a.manifest_dir / "parents.csv"
    crop_manifest_path = ref_a.manifest_dir / "reference_crops.csv"
    if not crop_manifest_path.exists():
        raise FileNotFoundError("Run `ising-mc make-crops` before parent-size-check")
    with crop_manifest_path.open(newline="", encoding="utf-8") as handle:
        reference_crops = list(csv.DictReader(handle))
    parents = read_parent_manifest(manifest_path)
    check_parents = [row for row in parents if row["split"] == check.split]
    if not check_parents:
        raise FileNotFoundError("Generate critical_L1024_parent_size_check.yaml first")

    import yaml

    spec_file = (
        Path(crop_spec_path).resolve()
        if crop_spec_path
        else config_directory / "crops.yaml"
    )
    spec = yaml.safe_load(spec_file.read_bytes())
    base_seed = int(spec["base_seed"])
    results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"radii": np.arange(max_distance + 1)}

    for size in (128, 256):
        rows_a = [
            row
            for row in reference_crops
            if row["split"] == "reference_a" and int(row["crop_size"]) == size
        ]
        rows_b = [
            row
            for row in reference_crops
            if row["split"] == "reference_b" and int(row["crop_size"]) == size
        ]
        rows_1024: list[dict[str, str]] = []
        for parent in check_parents:
            parent_size = int(parent["parent_size"])
            seed = _crop_seed(
                base_seed,
                role_id=4,
                size=size,
                chain_id=int(parent["chain_id"]),
                parent_index=int(parent["index_in_chain"]),
                repeat=0,
            )
            generator = np.random.default_rng(seed)
            maximum = parent_size - size
            rows_1024.append(
                {
                    **parent,
                    "parent_id": parent["sample_id"],
                    "crop_size": str(size),
                    "crop_top": str(int(generator.integers(maximum + 1))),
                    "crop_left": str(int(generator.integers(maximum + 1))),
                }
            )
        mean_a, chains_a = _mean_by_chain(root, rows_a, max_distance)
        mean_b, chains_b = _mean_by_chain(root, rows_b, max_distance)
        mean_1024, chains_1024 = _mean_by_chain(root, rows_1024, max_distance)
        evaluation = slice(1, max_distance + 1)
        natural_rms = float(
            np.sqrt(np.mean((mean_a[evaluation] - mean_b[evaluation]) ** 2))
        )
        parent_rms = float(
            np.sqrt(np.mean((mean_a[evaluation] - mean_1024[evaluation]) ** 2))
        )
        results[str(size)] = {
            "mc_mc_natural_rms": natural_rms,
            "parent_512_1024_rms": parent_rms,
            "ratio": parent_rms / natural_rms if natural_rms > 0 else float("inf"),
            "passed": parent_rms <= natural_rms,
            "n_512_reference_a": len(rows_a),
            "n_512_reference_b": len(rows_b),
            "n_1024": len(rows_1024),
        }
        arrays[f"w{size}__mean_512_a"] = mean_a
        arrays[f"w{size}__mean_512_b"] = mean_b
        arrays[f"w{size}__mean_1024"] = mean_1024
        arrays[f"w{size}__chains_512_a"] = chains_a
        arrays[f"w{size}__chains_512_b"] = chains_b
        arrays[f"w{size}__chains_1024"] = chains_1024

    overall = all(item["passed"] for item in results.values())
    arrays["summary"] = np.asarray(str({"passed": overall, "sizes": results}))
    atomic_savez(ref_a.report_dir / "parent_size_check.npz", **arrays)
    lines = [
        "# 父场尺寸效应检查",
        "",
        f"总体状态：**{'PASS' if overall else 'FAIL'}**",
        "",
        "判据：512 与 1024 父场 crop 的 RMS 差异不超过 reference_a 与 reference_b 的 MC–MC 自然差异。",
        "",
        "| crop | MC–MC RMS | 512–1024 RMS | ratio | 状态 |",
        "|:--|--:|--:|--:|:--|",
    ]
    for size, result in results.items():
        lines.append(
            f"| {size} | {result['mc_mc_natural_rms']:.6g} | {result['parent_512_1024_rms']:.6g} | "
            f"{result['ratio']:.3f} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
    (ref_a.report_dir / "parent_size_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {"passed": overall, "sizes": results}
