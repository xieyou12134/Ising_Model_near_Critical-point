from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import load_config
from .io_utils import atomic_savez
from .parent_size import open_axial_correlation


class _CropReader:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.cache: dict[str, np.ndarray] = {}

    def get(self, row: dict[str, str]) -> np.ndarray:
        shard_key = row["shard_path"]
        if shard_key not in self.cache:
            shard_path = Path(shard_key)
            if not shard_path.is_absolute():
                shard_path = self.root / shard_path
            self.cache[shard_key] = np.load(shard_path, mmap_mode="r")
        parents = self.cache[shard_key]
        top = int(row["crop_top"])
        left = int(row["crop_left"])
        size = int(row["crop_size"])
        return np.asarray(
            parents[int(row["index_in_chain"]), top : top + size, left : left + size]
        )


def correlation_permutation_pvalue(
    chains_a: np.ndarray,
    chains_b: np.ndarray,
    *,
    seed: int = 2026080961,
    max_permutations: int = 100_000,
) -> tuple[float, float, int]:
    """Compare two sets of chain-mean correlation curves without SciPy.

    The returned tuple is ``(observed_rms, p_value, permutations)``. Small
    chain sets use the exact balanced-label permutation distribution; larger
    sets use a deterministic Monte Carlo approximation.
    """

    first = np.asarray(chains_a, dtype=np.float64)
    second = np.asarray(chains_b, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("Chain curves must be two rank-2 arrays of equal width")
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Each reference split must contain at least two chains")

    def difference(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sqrt(np.mean((left.mean(axis=0) - right.mean(axis=0)) ** 2)))

    observed = difference(first, second)
    pooled = np.concatenate((first, second), axis=0)
    n_first = len(first)
    n_total = len(pooled)
    exact_count = math.comb(n_total, n_first)
    exceedances = 0

    if exact_count <= max_permutations:
        permutations = exact_count
        all_indices = np.arange(n_total)
        for selected_tuple in itertools.combinations(range(n_total), n_first):
            selected = np.asarray(selected_tuple, dtype=np.int64)
            complement = np.setdiff1d(all_indices, selected, assume_unique=True)
            exceedances += difference(pooled[selected], pooled[complement]) >= (
                observed - 1e-15
            )
        p_value = exceedances / permutations
    else:
        permutations = max_permutations
        generator = np.random.default_rng(seed)
        for _ in range(permutations):
            shuffled = generator.permutation(n_total)
            exceedances += difference(
                pooled[shuffled[:n_first]], pooled[shuffled[n_first:]]
            ) >= (observed - 1e-15)
        p_value = (exceedances + 1) / (permutations + 1)
    return observed, float(p_value), permutations


def _summarize_split(
    reader: _CropReader,
    rows: list[dict[str, str]],
    max_distance: int,
) -> dict[str, Any]:
    by_chain: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "correlation": np.zeros(max_distance + 1, dtype=np.float64),
            "scalars": np.zeros(4, dtype=np.float64),
        }
    )
    for row in rows:
        crop = reader.get(row)
        if crop.shape != (int(row["crop_size"]), int(row["crop_size"])):
            raise ValueError(f"Malformed crop: {row['crop_id']}")
        if not np.all((crop == -1) | (crop == 1)):
            raise ValueError(
                f"Crop contains values outside {{-1,+1}}: {row['crop_id']}"
            )
        magnetization = float(np.mean(crop, dtype=np.float64))
        correlation = open_axial_correlation(crop, max_distance)
        scalars = np.asarray(
            [-2.0 * correlation[1], magnetization, magnetization**2, magnetization**4]
        )
        accumulator = by_chain[int(row["chain_id"])]
        accumulator["count"] += 1
        accumulator["correlation"] += correlation
        accumulator["scalars"] += scalars

    if len(by_chain) < 2:
        raise ValueError("Each reference split must contain at least two chains")
    chain_ids = sorted(by_chain)
    chain_correlations = np.stack(
        [by_chain[key]["correlation"] / by_chain[key]["count"] for key in chain_ids]
    )
    chain_scalars = np.stack(
        [by_chain[key]["scalars"] / by_chain[key]["count"] for key in chain_ids]
    )
    return {
        "n_crops": len(rows),
        "chain_ids": chain_ids,
        "chain_correlations": chain_correlations,
        "chain_scalars": chain_scalars,
    }


def run_fixed_background_check(
    config_dir: str | Path,
    crop_spec_path: str | Path | None = None,
    distance_fraction: float = 0.25,
    significance: float = 0.05,
    magnetization_z_limit: float = 3.0,
) -> dict[str, Any]:
    """Evaluate training crop sizes within a fixed L=512 background ensemble."""

    if not 0.0 < distance_fraction <= 0.5:
        raise ValueError("distance_fraction must lie in (0, 0.5]")
    if not 0.0 < significance < 1.0:
        raise ValueError("significance must lie in (0, 1)")

    config_directory = Path(config_dir).resolve()
    ref_a = load_config(config_directory / "critical_L512_reference_a.yaml")
    ref_b = load_config(config_directory / "critical_L512_reference_b.yaml")
    if (
        ref_a.parent_size != 512
        or ref_b.parent_size != 512
        or ref_a.monte_carlo_root != ref_b.monte_carlo_root
        or ref_a.manifest_dir != ref_b.manifest_dir
    ):
        raise ValueError("Reference splits must share one fixed L=512 background")

    spec_path = (
        Path(crop_spec_path).resolve()
        if crop_spec_path
        else config_directory / "crops.yaml"
    )
    spec = yaml.safe_load(spec_path.read_bytes())
    training_sizes = tuple(sorted({int(value) for value in spec["training_sizes"]}))
    reference_sizes = {int(value) for value in spec["reference_sizes"]}
    missing = sorted(set(training_sizes).difference(reference_sizes))
    if missing:
        raise ValueError(f"Reference crop manifest omits training sizes: {missing}")

    crop_manifest = ref_a.manifest_dir / "reference_crops.csv"
    if not crop_manifest.exists():
        raise FileNotFoundError(
            "Run `ising-mc make-crops` before fixed-background-check"
        )
    with crop_manifest.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))

    reader = _CropReader(ref_a.monte_carlo_root)
    results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for size in training_sizes:
        max_distance = max(1, min(size - 1, int(size * distance_fraction)))
        rows_a = [
            row
            for row in all_rows
            if row["split"] == "reference_a" and int(row["crop_size"]) == size
        ]
        rows_b = [
            row
            for row in all_rows
            if row["split"] == "reference_b" and int(row["crop_size"]) == size
        ]
        if not rows_a or not rows_b:
            raise ValueError(f"Missing frozen reference crops for width {size}")
        summary_a = _summarize_split(reader, rows_a, max_distance)
        summary_b = _summarize_split(reader, rows_b, max_distance)
        curves_a = summary_a["chain_correlations"][:, 1:]
        curves_b = summary_b["chain_correlations"][:, 1:]
        correlation_rms, p_value, permutations = correlation_permutation_pvalue(
            curves_a, curves_b
        )

        chain_scalars = np.concatenate(
            (summary_a["chain_scalars"], summary_b["chain_scalars"]), axis=0
        )
        mean_scalars = chain_scalars.mean(axis=0)
        magnetization_se = float(
            np.std(chain_scalars[:, 1], ddof=1) / np.sqrt(len(chain_scalars))
        )
        mean_magnetization = float(mean_scalars[1])
        magnetization_z = (
            abs(mean_magnetization) / magnetization_se
            if magnetization_se > 0.0
            else (0.0 if mean_magnetization == 0.0 else float("inf"))
        )
        binder_u4 = float(1.0 - mean_scalars[3] / (3.0 * mean_scalars[2] ** 2))
        checks = {
            "reference_exchangeability": p_value >= significance,
            "magnetization_zero": magnetization_z <= magnetization_z_limit,
        }
        passed = all(checks.values())
        results[str(size)] = {
            "passed": passed,
            "n_reference_a": summary_a["n_crops"],
            "n_reference_b": summary_b["n_crops"],
            "n_chains_a": len(summary_a["chain_ids"]),
            "n_chains_b": len(summary_b["chain_ids"]),
            "max_distance": max_distance,
            "mean_energy": float(mean_scalars[0]),
            "mean_magnetization": mean_magnetization,
            "magnetization_se": magnetization_se,
            "magnetization_z": float(magnetization_z),
            "binder_u4": binder_u4,
            "reference_correlation_rms": correlation_rms,
            "permutation_p_value": p_value,
            "permutations": permutations,
            "checks": checks,
        }
        arrays[f"w{size}__radii"] = np.arange(max_distance + 1)
        arrays[f"w{size}__chains_a"] = summary_a["chain_correlations"]
        arrays[f"w{size}__chains_b"] = summary_b["chain_correlations"]
        arrays[f"w{size}__scalars_a"] = summary_a["chain_scalars"]
        arrays[f"w{size}__scalars_b"] = summary_b["chain_scalars"]

    overall = all(item["passed"] for item in results.values())
    output = {"passed": overall, "parent_size": 512, "sizes": results}
    arrays["summary"] = np.asarray(str(output))
    atomic_savez(ref_a.report_dir / "fixed_background_check.npz", **arrays)

    lines = [
        "# 固定 $L=512$ 背景 crop 评估",
        "",
        f"总体状态：**{'PASS' if overall else 'FAIL'}**",
        "",
        "本检查不比较不同父场尺寸，而是检查两个独立 $L=512$ reference split 的局部关联是否可交换，并检查零场自旋翻转对称性。",
        "",
        "| crop | 每 split 样本 | 评价距离 | 能量 | 磁化 | Binder $U_4$ | 关联 RMS | permutation $p$ | 状态 |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    for size, result in results.items():
        lines.append(
            f"| {size} | {result['n_reference_a']} | $r\\leq{result['max_distance']}$ | "
            f"{result['mean_energy']:.6f} | {result['mean_magnetization']:.6f} | "
            f"{result['binder_u4']:.5f} | {result['reference_correlation_rms']:.6g} | "
            f"{result['permutation_p_value']:.4f} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"门禁：关联曲线的平衡标签置换检验要求 $p\\geq{significance:g}$；合并链的平均磁化要求与 $0$ 的偏差不超过 {magnetization_z_limit:g} 个链级标准误。",
            "",
            "统计独立单位始终是 Monte Carlo 链；同一父构型的多个 crop 不会被当作额外独立样本。",
        ]
    )
    (ref_a.report_dir / "fixed_background_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return output
