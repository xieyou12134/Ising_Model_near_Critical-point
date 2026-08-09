from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig, load_config
from .io_utils import atomic_savez, atomic_write_csv, sha256_file

CHAIN_FIELDS = [
    "split",
    "chain_id",
    "initial_state",
    "n_samples",
    "mean_energy",
    "mean_abs_m",
    "mean_m2",
    "mean_m4",
    "mean_skmin",
    "tau_energy",
    "tau_abs_m",
    "tau_m2",
    "tau_m4",
    "tau_skmin",
    "ess_energy",
    "ess_abs_m",
    "ess_m2",
    "ess_m4",
    "ess_skmin",
    "min_ess",
]


def integrated_autocorrelation_time(series: np.ndarray) -> float:
    """Geyer initial-positive-sequence estimate with tau >= 1/2."""
    values = np.asarray(series, dtype=np.float64)
    n_values = values.size
    if n_values < 2:
        return 0.5
    values = values - np.mean(values)
    variance = float(np.dot(values, values) / n_values)
    if variance <= np.finfo(np.float64).tiny:
        return 0.5
    fft_size = 1 << (2 * n_values - 1).bit_length()
    transformed = np.fft.rfft(values, n=fft_size)
    autocovariance = np.fft.irfft(transformed * np.conjugate(transformed), n=fft_size)[
        :n_values
    ]
    autocovariance /= np.arange(n_values, 0, -1)
    autocorrelation = autocovariance / autocovariance[0]

    tau = 0.5
    lag = 1
    while lag + 1 < n_values:
        pair_sum = float(autocorrelation[lag] + autocorrelation[lag + 1])
        if pair_sum <= 0:
            break
        tau += pair_sum
        lag += 2
    if lag < n_values and autocorrelation[lag] > 0:
        tau += float(autocorrelation[lag])
    return max(0.5, tau)


def effective_sample_size(series: np.ndarray) -> tuple[float, float]:
    tau = integrated_autocorrelation_time(series)
    return tau, min(float(len(series)), float(len(series)) / (2.0 * tau))


def split_rhat(chains: Iterable[np.ndarray]) -> float:
    split_chains: list[np.ndarray] = []
    for chain in chains:
        values = np.asarray(chain, dtype=np.float64)
        half = values.size // 2
        if half >= 2:
            split_chains.extend((values[:half], values[-half:]))
    if len(split_chains) < 2:
        return float("nan")
    matrix = np.stack(split_chains)
    length = matrix.shape[1]
    within = float(np.mean(np.var(matrix, axis=1, ddof=1)))
    between = float(length * np.var(np.mean(matrix, axis=1), ddof=1))
    if within == 0:
        return 1.0 if between == 0 else float("inf")
    variance = ((length - 1.0) / length) * within + between / length
    return math.sqrt(max(0.0, variance / within))


def low_mode_structure(spins: np.ndarray, phase: np.ndarray) -> float:
    n_sites = spins.size
    mode_x = np.dot(np.sum(spins, axis=0, dtype=np.int64), phase)
    mode_y = np.dot(np.sum(spins, axis=1, dtype=np.int64), phase)
    return float((abs(mode_x) ** 2 + abs(mode_y) ** 2) / (2.0 * n_sites))


def chain_bootstrap_mean_ci(
    chain_values: np.ndarray, seed: int, n_resamples: int = 5_000
) -> np.ndarray:
    """Return a 95% interval after resampling whole independent chains."""
    values = np.asarray(chain_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    n_chains = values.shape[0]
    if n_chains < 2:
        return np.stack((values[0], values[0]))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, n_chains, size=(n_resamples, n_chains))
    replicates = np.mean(values[indices], axis=1)
    return np.quantile(replicates, (0.025, 0.975), axis=0)


def _chain_files(config: RunConfig, chain_id: int) -> tuple[Path, Path, Path]:
    stem = config.split_dir / f"chain_{chain_id:03d}"
    return (
        stem.with_suffix(".npy"),
        stem.with_suffix(".metrics.npz"),
        stem.with_suffix(".meta.json"),
    )


def _gibbs_counts(
    spins: np.ndarray, counts: np.ndarray, plus_counts: np.ndarray
) -> None:
    neighbors = (
        np.roll(spins, 1, axis=0)
        + np.roll(spins, -1, axis=0)
        + np.roll(spins, 1, axis=1)
        + np.roll(spins, -1, axis=1)
    )
    for index, q_value in enumerate((-4, -2, 0, 2, 4)):
        mask = neighbors == q_value
        counts[index] += int(np.count_nonzero(mask))
        plus_counts[index] += int(np.count_nonzero(mask & (spins == 1)))


def diagnose_split(
    config: RunConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    size_l = config.parent_size
    n_sites = config.n_sites
    phase = np.exp(2j * np.pi * np.arange(size_l) / size_l)
    structure_sum = np.zeros((size_l, size_l), dtype=np.float64)
    chain_structure_means: list[np.ndarray] = []
    gibbs_counts = np.zeros(5, dtype=np.int64)
    gibbs_plus = np.zeros(5, dtype=np.int64)
    total_samples = 0
    chain_rows: list[dict[str, Any]] = []
    energy_chains: list[np.ndarray] = []
    magnetization_chains: list[np.ndarray] = []
    chain_means_energy: list[float] = []
    chain_means_magnetization: list[float] = []
    seen_seeds: set[int] = set()

    for chain_id in range(config.n_chains):
        shard_path, metrics_path, meta_path = _chain_files(config, chain_id)
        if not (shard_path.exists() and metrics_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Incomplete chain {config.split}/{chain_id:03d}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata["config_sha256"] != config.source_sha256:
            raise RuntimeError(f"Config checksum mismatch for {meta_path}")
        if metadata["sha256"] != sha256_file(shard_path):
            raise RuntimeError(f"Shard checksum mismatch for {shard_path}")
        seed = int(metadata["chain_seed"])
        if seed in seen_seeds:
            raise RuntimeError(f"Repeated seed within split {config.split}: {seed}")
        seen_seeds.add(seed)

        parents = np.load(shard_path, mmap_mode="r")
        expected_shape = (config.n_samples_per_chain, size_l, size_l)
        if parents.shape != expected_shape or parents.dtype != np.int8:
            raise RuntimeError(
                f"Invalid array contract for {shard_path}: {parents.shape}, {parents.dtype}"
            )
        with np.load(metrics_path) as metrics:
            energy = np.asarray(metrics["energy"], dtype=np.float64)
            magnetization = np.asarray(metrics["magnetization"], dtype=np.float64)
        if (
            energy.shape != (config.n_samples_per_chain,)
            or magnetization.shape != energy.shape
        ):
            raise RuntimeError(f"Invalid metrics contract for {metrics_path}")

        skmin = np.empty(config.n_samples_per_chain, dtype=np.float64)
        chain_structure_sum = np.zeros_like(structure_sum)
        for sample_index in range(config.n_samples_per_chain):
            spins = np.asarray(parents[sample_index])
            if not np.all((spins == -1) | (spins == 1)):
                raise RuntimeError(
                    f"Invalid spin value in {shard_path}, sample {sample_index}"
                )
            observed_energy = -float(
                np.mean(spins * np.roll(spins, -1, axis=0))
                + np.mean(spins * np.roll(spins, -1, axis=1))
            )
            observed_magnetization = float(np.mean(spins))
            if abs(observed_energy - energy[sample_index]) > 1e-12:
                raise RuntimeError(
                    f"Stored energy mismatch in {shard_path}, sample {sample_index}"
                )
            if abs(observed_magnetization - magnetization[sample_index]) > 1e-12:
                raise RuntimeError(
                    f"Stored magnetization mismatch in {shard_path}, sample {sample_index}"
                )
            transformed = np.fft.fft2(spins)
            sample_structure = (transformed.real**2 + transformed.imag**2) / n_sites
            structure_sum += sample_structure
            chain_structure_sum += sample_structure
            skmin[sample_index] = low_mode_structure(spins, phase)
            _gibbs_counts(spins, gibbs_counts, gibbs_plus)
        total_samples += config.n_samples_per_chain
        chain_structure_means.append(chain_structure_sum / config.n_samples_per_chain)

        abs_m = np.abs(magnetization)
        series = {
            "energy": energy,
            "abs_m": abs_m,
            "m2": magnetization**2,
            "m4": magnetization**4,
            "skmin": skmin,
        }
        tau_ess = {
            name: effective_sample_size(values) for name, values in series.items()
        }
        min_ess = min(pair[1] for pair in tau_ess.values())
        chain_rows.append(
            {
                "split": config.split,
                "chain_id": chain_id,
                "initial_state": metadata["initial_state"],
                "n_samples": config.n_samples_per_chain,
                "mean_energy": float(np.mean(energy)),
                "mean_abs_m": float(np.mean(abs_m)),
                "mean_m2": float(np.mean(series["m2"])),
                "mean_m4": float(np.mean(series["m4"])),
                "mean_skmin": float(np.mean(skmin)),
                "tau_energy": tau_ess["energy"][0],
                "tau_abs_m": tau_ess["abs_m"][0],
                "tau_m2": tau_ess["m2"][0],
                "tau_m4": tau_ess["m4"][0],
                "tau_skmin": tau_ess["skmin"][0],
                "ess_energy": tau_ess["energy"][1],
                "ess_abs_m": tau_ess["abs_m"][1],
                "ess_m2": tau_ess["m2"][1],
                "ess_m4": tau_ess["m4"][1],
                "ess_skmin": tau_ess["skmin"][1],
                "min_ess": min_ess,
            }
        )
        energy_chains.append(energy)
        magnetization_chains.append(magnetization)
        chain_means_energy.append(float(np.mean(energy)))
        chain_means_magnetization.append(float(np.mean(magnetization)))

    mean_structure = structure_sum / total_samples
    mean_correlation = np.fft.ifft2(mean_structure).real
    mean_energy = float(np.mean(np.concatenate(energy_chains)))
    all_magnetization = np.concatenate(magnetization_chains)
    mean_abs_m = float(np.mean(np.abs(all_magnetization)))
    mean_m2 = float(np.mean(all_magnetization**2))
    mean_m4 = float(np.mean(all_magnetization**4))
    binder_u4 = 1.0 - mean_m4 / (3.0 * mean_m2**2)
    mean_skmin = float(np.mean([row["mean_skmin"] for row in chain_rows]))
    s0_expected = n_sites * mean_m2
    xi2 = (1.0 / (2.0 * math.sin(math.pi / size_l))) * math.sqrt(
        max(s0_expected / mean_skmin - 1.0, 0.0)
    )
    energy_from_g = -float(mean_correlation[1, 0] + mean_correlation[0, 1])
    invariants = {
        "g0_error": abs(float(mean_correlation[0, 0]) - 1.0),
        "parseval_error": abs(float(np.mean(mean_structure)) - 1.0),
        "energy_correlation_error": abs(energy_from_g - mean_energy),
        "s0_error": abs(float(mean_structure[0, 0]) - s0_expected),
        "fft_pair_error": float(
            np.max(np.abs(np.fft.fft2(mean_correlation) - mean_structure))
        ),
    }

    q_values = np.asarray([-4, -2, 0, 2, 4], dtype=np.int8)
    empirical = np.divide(
        gibbs_plus,
        gibbs_counts,
        out=np.full(5, np.nan, dtype=np.float64),
        where=gibbs_counts > 0,
    )
    exact = 1.0 / (1.0 + np.exp(-2.0 * config.beta * q_values))
    eligible = gibbs_counts >= 500
    gibbs_max_error = (
        float(np.max(np.abs(empirical[eligible] - exact[eligible])))
        if np.any(eligible)
        else math.inf
    )

    chain_energy_array = np.asarray(chain_means_energy)
    energy_mcse = (
        float(np.std(chain_energy_array, ddof=1) / math.sqrt(config.n_chains))
        if config.n_chains > 1
        else float(np.std(energy_chains[0], ddof=1) / math.sqrt(len(energy_chains[0])))
    )
    chain_m_array = np.asarray(chain_means_magnetization)
    m_chain_se = (
        float(np.std(chain_m_array, ddof=1) / math.sqrt(config.n_chains))
        if config.n_chains > 1
        else float(
            np.std(magnetization_chains[0], ddof=1)
            / math.sqrt(len(magnetization_chains[0]))
        )
    )
    rhat_energy = split_rhat(energy_chains)
    rhat_abs_m = split_rhat([np.abs(values) for values in magnetization_chains])
    min_chain_ess = min(float(row["min_ess"]) for row in chain_rows)

    max_distance = min(64, size_l // 4)
    radii = np.arange(1, max_distance + 1)
    axial = np.asarray(
        [
            (
                mean_correlation[0, radius]
                + mean_correlation[0, -radius]
                + mean_correlation[radius, 0]
                + mean_correlation[-radius, 0]
            )
            / 4.0
            for radius in radii
        ]
    )
    chain_correlations = np.stack(
        [np.fft.ifft2(value).real for value in chain_structure_means]
    )
    chain_axial = np.stack(
        [
            np.asarray(
                [
                    (
                        correlation[0, radius]
                        + correlation[0, -radius]
                        + correlation[radius, 0]
                        + correlation[-radius, 0]
                    )
                    / 4.0
                    for radius in radii
                ]
            )
            for correlation in chain_correlations
        ]
    )
    axial_ci95 = chain_bootstrap_mean_ci(chain_axial, config.base_seed + 701)
    energy_ci95 = chain_bootstrap_mean_ci(chain_energy_array, config.base_seed + 702)[
        :, 0
    ]
    magnetization_ci95 = chain_bootstrap_mean_ci(chain_m_array, config.base_seed + 703)[
        :, 0
    ]
    fit_mask = (radii >= min(8, max_distance)) & (axial > 0)
    fitted_exponent = (
        float(-np.polyfit(np.log(radii[fit_mask]), np.log(axial[fit_mask]), 1)[0])
        if np.count_nonzero(fit_mask) >= 3
        else float("nan")
    )

    checks = {
        "array_contract": True,
        "invariants": all(value <= 1e-10 for value in invariants.values()),
        "rhat": rhat_energy <= 1.05 and rhat_abs_m <= 1.05,
        "ess_min_30": min_chain_ess >= 30.0,
        "magnetization_zero": abs(float(np.mean(all_magnetization)))
        <= 3.0 * m_chain_se,
        "critical_energy": abs(mean_energy + math.sqrt(2.0))
        <= max(5.0 * energy_mcse, 0.02),
        "local_gibbs": gibbs_max_error <= 0.05,
    }
    summary = {
        "split": config.split,
        "n_chains": config.n_chains,
        "n_samples": total_samples,
        "mean_energy": mean_energy,
        "energy_mcse": energy_mcse,
        "energy_chain_bootstrap_ci95": energy_ci95.tolist(),
        "mean_magnetization": float(np.mean(all_magnetization)),
        "mean_abs_magnetization": mean_abs_m,
        "mean_m2": mean_m2,
        "mean_m4": mean_m4,
        "binder_u4": binder_u4,
        "mean_s0": s0_expected,
        "mean_skmin": mean_skmin,
        "xi2_over_l": xi2 / size_l,
        "magnetization_chain_se": m_chain_se,
        "magnetization_chain_bootstrap_ci95": magnetization_ci95.tolist(),
        "rhat_energy": rhat_energy,
        "rhat_abs_m": rhat_abs_m,
        "min_chain_ess": min_chain_ess,
        "target_ess_100_met": min_chain_ess >= 100.0,
        "gibbs_max_error": gibbs_max_error,
        "fitted_correlation_exponent": fitted_exponent,
        "invariants": invariants,
        "checks": checks,
        "passed": all(checks.values()),
    }
    arrays = {
        "correlation": mean_correlation,
        "structure_factor": mean_structure,
        "radii": radii,
        "axial_correlation": axial,
        "chain_axial_correlation": chain_axial,
        "axial_correlation_chain_bootstrap_ci95": axial_ci95,
        "gibbs_q": q_values,
        "gibbs_counts": gibbs_counts,
        "gibbs_empirical": empirical,
        "gibbs_exact": exact,
    }
    return chain_rows, summary, arrays


def diagnose_configs(config_paths: Iterable[str | Path]) -> dict[str, Any]:
    configs = [load_config(path) for path in config_paths]
    if not configs:
        raise ValueError("At least one config is required")
    report_dir = configs[0].report_dir
    if any(config.report_dir != report_dir for config in configs):
        raise ValueError("All configs must use the same report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    output_arrays: dict[str, np.ndarray] = {}
    all_seeds: set[int] = set()
    seeds_unique = True
    for config in configs:
        rows, summary, arrays = diagnose_split(config)
        all_rows.extend(rows)
        summaries[config.split] = summary
        for name, value in arrays.items():
            output_arrays[f"{config.split}__{name}"] = value
        for chain_id in range(config.n_chains):
            _, _, meta_path = _chain_files(config, chain_id)
            seed = int(json.loads(meta_path.read_text(encoding="utf-8"))["chain_seed"])
            if seed in all_seeds:
                seeds_unique = False
            all_seeds.add(seed)

    atomic_write_csv(report_dir / "chain_diagnostics.csv", CHAIN_FIELDS, all_rows)
    output_arrays["summary_json"] = np.asarray(json.dumps(summaries, sort_keys=True))
    atomic_savez(report_dir / "observables.npz", **output_arrays)

    overall_pass = seeds_unique and all(
        summary["passed"] for summary in summaries.values()
    )
    lines = [
        "# Monte Carlo 数据验收报告",
        "",
        f"- 总体状态：**{'PASS' if overall_pass else 'FAIL'}**",
        f"- 跨 split 链 seed 唯一：`{seeds_unique}`",
        "",
        "| split | samples | energy | Binder U4 | xi2/L | R-hat(e) | R-hat(|m|) | min ESS | Gibbs max error | 状态 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for split, summary in summaries.items():
        lines.append(
            f"| {split} | {summary['n_samples']} | {summary['mean_energy']:.8f} | "
            f"{summary['binder_u4']:.5f} | {summary['xi2_over_l']:.5f} | "
            f"{summary['rhat_energy']:.4f} | {summary['rhat_abs_m']:.4f} | "
            f"{summary['min_chain_ess']:.1f} | {summary['gibbs_max_error']:.5f} | "
            f"{'PASS' if summary['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "## 逐项检查", ""])
    for split, summary in summaries.items():
        lines.append(f"### {split}")
        lines.append("")
        for name, passed in summary["checks"].items():
            lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
        lines.append(
            f"- 中距离相关拟合指数（诊断项）：`{summary['fitted_correlation_exponent']:.4f}`"
        )
        lines.append(f"- 每链最小 ESS 达到目标100：`{summary['target_ess_100_met']}`")
        lines.append("")
    (report_dir / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    return {"passed": overall_pass, "seeds_unique": seeds_unique, "splits": summaries}
