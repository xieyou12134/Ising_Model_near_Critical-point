from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BETA_CRITICAL = 0.44068679350977147


@dataclass(frozen=True)
class DataSpec:
    root: Path
    train_manifest: Path
    train_split: str
    validation_manifest: Path | None
    validation_split: str
    widths: tuple[int, ...]
    batch_size: int
    seed: int


@dataclass(frozen=True)
class ModelSpec:
    dim: int
    depth: int
    heads: int
    mlp_ratio: float
    dropout: float
    qk_norm: bool
    rope_base: float
    condition_on_beta: bool
    fixed_beta: float

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads


@dataclass(frozen=True)
class ObjectiveSpec:
    t_min: float
    t_max: float


@dataclass(frozen=True)
class TrainingSpec:
    steps: int
    gradient_accumulation: int
    learning_rate: float
    adam_betas: tuple[float, float]
    weight_decay: float
    warmup_steps: int
    min_lr_ratio: float
    gradient_clip: float
    precision: str
    device: str
    log_every: int
    validate_every: int
    checkpoint_every: int


@dataclass(frozen=True)
class ValidationSpec:
    t_grid: tuple[float, ...]
    batches_per_width: int


@dataclass(frozen=True)
class ExperimentSpec:
    run_id: str
    output_dir: Path
    seed: int
    data: DataSpec
    model: ModelSpec
    objective: ObjectiveSpec
    training: TrainingSpec
    validation: ValidationSpec
    source_path: Path
    raw: dict[str, Any]


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def load_experiment(path: str | Path) -> ExperimentSpec:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Experiment config must be a mapping: {source}")

    base = source.parent
    run = dict(_required(raw, "run"))
    data_raw = dict(_required(raw, "data"))
    model_raw = dict(_required(raw, "model"))
    objective_raw = dict(_required(raw, "objective"))
    train_raw = dict(_required(raw, "training"))
    validation_raw = dict(_required(raw, "validation"))

    validation_manifest_raw = data_raw.get("validation_manifest")
    spec = ExperimentSpec(
        run_id=str(_required(run, "id")),
        output_dir=_resolve(str(_required(run, "output_dir")), base),
        seed=int(run.get("seed", 2026080900)),
        data=DataSpec(
            root=_resolve(str(_required(data_raw, "root")), base),
            train_manifest=_resolve(str(_required(data_raw, "train_manifest")), base),
            train_split=str(data_raw.get("train_split", "train")),
            validation_manifest=(
                _resolve(str(validation_manifest_raw), base)
                if validation_manifest_raw
                else None
            ),
            validation_split=str(data_raw.get("validation_split", "val")),
            widths=tuple(int(value) for value in _required(data_raw, "widths")),
            batch_size=int(_required(data_raw, "batch_size")),
            seed=int(data_raw.get("seed", run.get("seed", 2026080900))),
        ),
        model=ModelSpec(
            dim=int(_required(model_raw, "dim")),
            depth=int(_required(model_raw, "depth")),
            heads=int(_required(model_raw, "heads")),
            mlp_ratio=float(model_raw.get("mlp_ratio", 4.0)),
            dropout=float(model_raw.get("dropout", 0.0)),
            qk_norm=bool(model_raw.get("qk_norm", True)),
            rope_base=float(model_raw.get("rope_base", 10_000.0)),
            condition_on_beta=bool(model_raw.get("condition_on_beta", False)),
            fixed_beta=float(model_raw.get("fixed_beta", BETA_CRITICAL)),
        ),
        objective=ObjectiveSpec(
            t_min=float(objective_raw.get("t_min", 0.2)),
            t_max=float(objective_raw.get("t_max", 1.0)),
        ),
        training=TrainingSpec(
            steps=int(_required(train_raw, "steps")),
            gradient_accumulation=int(train_raw.get("gradient_accumulation", 1)),
            learning_rate=float(train_raw.get("learning_rate", 2e-4)),
            adam_betas=tuple(
                float(value) for value in train_raw.get("adam_betas", (0.9, 0.95))
            ),
            weight_decay=float(train_raw.get("weight_decay", 0.01)),
            warmup_steps=int(train_raw.get("warmup_steps", 0)),
            min_lr_ratio=float(train_raw.get("min_lr_ratio", 0.1)),
            gradient_clip=float(train_raw.get("gradient_clip", 1.0)),
            precision=str(train_raw.get("precision", "bf16")),
            device=str(train_raw.get("device", "auto")),
            log_every=int(train_raw.get("log_every", 10)),
            validate_every=int(train_raw.get("validate_every", 500)),
            checkpoint_every=int(train_raw.get("checkpoint_every", 1_000)),
        ),
        validation=ValidationSpec(
            t_grid=tuple(float(value) for value in validation_raw["t_grid"]),
            batches_per_width=int(validation_raw.get("batches_per_width", 2)),
        ),
        source_path=source,
        raw=raw,
    )
    validate_experiment(spec)
    return spec


def validate_experiment(spec: ExperimentSpec) -> None:
    if not spec.run_id:
        raise ValueError("run.id cannot be empty")
    if not spec.data.widths or any(width <= 0 for width in spec.data.widths):
        raise ValueError("data.widths must contain positive integers")
    if spec.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive")
    if spec.model.dim <= 0 or spec.model.depth <= 0 or spec.model.heads <= 0:
        raise ValueError("model dim, depth, and heads must be positive")
    if spec.model.dim % spec.model.heads:
        raise ValueError("model.dim must be divisible by model.heads")
    if spec.model.head_dim % 2:
        raise ValueError("attention head dimension must be even for RoPE")
    if not 0.0 <= spec.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if spec.model.fixed_beta <= 0:
        raise ValueError("model.fixed_beta must be positive")
    if not 0.0 < spec.objective.t_min <= spec.objective.t_max <= 1.0:
        raise ValueError("objective requires 0 < t_min <= t_max <= 1")
    if spec.training.steps <= 0 or spec.training.gradient_accumulation <= 0:
        raise ValueError("training steps and gradient accumulation must be positive")
    if len(spec.training.adam_betas) != 2:
        raise ValueError("training.adam_betas must contain exactly two values")
    if spec.training.precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("training.precision must be fp32, bf16, or fp16")
    if any(
        not spec.objective.t_min <= value <= spec.objective.t_max
        for value in spec.validation.t_grid
    ):
        raise ValueError("validation.t_grid values must lie within objective bounds")
