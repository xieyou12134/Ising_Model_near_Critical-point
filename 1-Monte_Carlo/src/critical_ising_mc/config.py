from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SPLITS = {
    "train",
    "val",
    "reference_a",
    "reference_b",
    "parent_size_check",
    "smoke",
}


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    split: str
    split_id: int
    beta: float
    parent_size: int
    n_chains: int
    n_samples_per_chain: int
    adaptation_sweeps: float
    pilot_cluster_steps: int
    pilot_rounds: int
    burnin_sweeps: float
    sweeps_between_samples: float
    base_seed: int
    dtype: str
    output_root: Path
    manifest_dir: Path
    report_dir: Path
    source_path: Path
    source_sha256: str
    monte_carlo_root: Path

    @property
    def split_dir(self) -> Path:
        return self.output_root / self.split

    @property
    def n_sites(self) -> int:
        return self.parent_size * self.parent_size

    @property
    def pilot_flips(self) -> int:
        return self.pilot_cluster_steps * self.pilot_rounds

    def metadata(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "split": self.split,
            "split_id": self.split_id,
            "beta": self.beta,
            "parent_size": self.parent_size,
            "n_chains": self.n_chains,
            "n_samples_per_chain": self.n_samples_per_chain,
            "adaptation_sweeps": self.adaptation_sweeps,
            "pilot_cluster_steps": self.pilot_cluster_steps,
            "pilot_rounds": self.pilot_rounds,
            "burnin_sweeps": self.burnin_sweeps,
            "sweeps_between_samples": self.sweeps_between_samples,
            "base_seed": self.base_seed,
            "dtype": self.dtype,
            "output_root": str(self.output_root),
            "manifest_dir": str(self.manifest_dir),
            "report_dir": str(self.report_dir),
            "config_sha256": self.source_sha256,
        }


def _resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def load_config(path: str | Path) -> RunConfig:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    values = yaml.safe_load(raw)
    if not isinstance(values, dict):
        raise TypeError(f"Config must contain a YAML mapping: {source}")

    required = {
        "run_name",
        "split",
        "split_id",
        "beta",
        "parent_size",
        "n_chains",
        "n_samples_per_chain",
        "adaptation_sweeps",
        "pilot_cluster_steps",
        "pilot_rounds",
        "burnin_sweeps",
        "sweeps_between_samples",
        "base_seed",
        "dtype",
        "output_root",
        "manifest_dir",
        "report_dir",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Missing config keys in {source}: {', '.join(missing)}")

    config_dir = source.parent
    monte_carlo_root = config_dir.parent.resolve()
    config = RunConfig(
        run_name=str(values["run_name"]),
        split=str(values["split"]),
        split_id=int(values["split_id"]),
        beta=float(values["beta"]),
        parent_size=int(values["parent_size"]),
        n_chains=int(values["n_chains"]),
        n_samples_per_chain=int(values["n_samples_per_chain"]),
        adaptation_sweeps=float(values["adaptation_sweeps"]),
        pilot_cluster_steps=int(values["pilot_cluster_steps"]),
        pilot_rounds=int(values["pilot_rounds"]),
        burnin_sweeps=float(values["burnin_sweeps"]),
        sweeps_between_samples=float(values["sweeps_between_samples"]),
        base_seed=int(values["base_seed"]),
        dtype=str(values["dtype"]),
        output_root=_resolve_path(str(values["output_root"]), config_dir),
        manifest_dir=_resolve_path(str(values["manifest_dir"]), config_dir),
        report_dir=_resolve_path(str(values["report_dir"]), config_dir),
        source_path=source,
        source_sha256=sha256(raw).hexdigest(),
        monte_carlo_root=monte_carlo_root,
    )
    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    if config.split not in ALLOWED_SPLITS:
        raise ValueError(f"Unsupported split {config.split!r}")
    if config.dtype != "int8":
        raise ValueError("Only dtype=int8 is supported by the data contract")
    if config.beta <= 0:
        raise ValueError("beta must be positive")
    if config.base_seed < 0 or config.split_id < 0:
        raise ValueError("base_seed and split_id must be non-negative")
    if config.parent_size < 2:
        raise ValueError("parent_size must be at least 2")
    integer_positive = {
        "n_chains": config.n_chains,
        "n_samples_per_chain": config.n_samples_per_chain,
        "pilot_cluster_steps": config.pilot_cluster_steps,
        "pilot_rounds": config.pilot_rounds,
    }
    for name, value in integer_positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("adaptation_sweeps", "burnin_sweeps", "sweeps_between_samples"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
