from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    atomic_write_json,
    atomic_write_text,
    prepare_run_directory,
    sha256_file,
)
from .data import FixedCropSource, ReplayableCropSource
from .model import IsingDiffusionModel
from .objective import AbsorbingDiffusionObjective
from .observables import summarize
from .sampler import sample_absorbing
from .spec import ExperimentSpec, load_experiment
from .system import IsingDiffusionSystem
from .trainer import Trainer


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _build_model(spec: ExperimentSpec) -> IsingDiffusionModel:
    model = spec.model
    return IsingDiffusionModel(
        dim=model.dim,
        depth=model.depth,
        heads=model.heads,
        mlp_ratio=model.mlp_ratio,
        dropout=model.dropout,
        qk_norm=model.qk_norm,
        rope_base=model.rope_base,
        condition_on_beta=model.condition_on_beta,
        fixed_beta=model.fixed_beta,
    )


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train critical Ising diffusion model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", help="Override config device, e.g. cuda:0 or cpu")
    parser.add_argument("--output", help="Override run.output_dir")
    args = parser.parse_args(argv)

    spec = load_experiment(args.config)
    if args.output:
        spec = replace(spec, output_dir=Path(args.output).expanduser().resolve())
    device = _device(args.device or spec.training.device)
    existing_last = spec.output_dir / "checkpoints" / "last.pt"
    if existing_last.exists() and not args.resume:
        raise FileExistsError(
            f"Run output already contains {existing_last}. Use --resume or choose a new run.id/output_dir."
        )
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    if not spec.data.train_manifest.exists():
        raise FileNotFoundError(
            f"Training manifest does not exist: {spec.data.train_manifest}. "
            "Generate Monte Carlo data first."
        )
    train_source = ReplayableCropSource(
        spec.data.train_manifest,
        spec.data.root,
        spec.data.train_split,
        spec.data.widths,
        spec.data.batch_size,
        spec.data.seed,
    )
    if spec.data.validation_manifest is not None:
        if not spec.data.validation_manifest.exists():
            raise FileNotFoundError(
                f"Fixed validation manifest is missing: {spec.data.validation_manifest}"
            )
        validation_source: FixedCropSource | ReplayableCropSource = FixedCropSource(
            spec.data.validation_manifest,
            spec.data.root,
            spec.data.validation_split,
            spec.data.batch_size,
            spec.data.seed + 1,
        )
    else:
        validation_source = ReplayableCropSource(
            spec.data.train_manifest,
            spec.data.root,
            spec.data.validation_split,
            spec.data.widths,
            spec.data.batch_size,
            spec.data.seed + 1,
        )

    model = _build_model(spec)
    objective = AbsorbingDiffusionObjective(spec.objective.t_min, spec.objective.t_max)
    system = IsingDiffusionSystem(model, objective)
    repository_root = spec.source_path.parent.parent.parent
    prepare_run_directory(
        spec.output_dir,
        spec.raw,
        spec.data.train_manifest,
        model.architecture_facts(),
        repository_root,
    )
    trainer = Trainer(spec, system, train_source, validation_source, device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed_from={Path(args.resume).resolve()} step={trainer.global_step}")
    print(
        f"run_id={spec.run_id} device={device} beta_mode="
        f"{'conditioned' if spec.model.condition_on_beta else 'fixed'} "
        f"fixed_beta={spec.model.fixed_beta:.15f}",
        flush=True,
    )
    trainer.fit()
    print(f"training_complete step={trainer.global_step}", flush=True)
    return 0


def sample_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample an Ising diffusion checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    spec = load_experiment(args.config)
    device = _device(args.device or spec.training.device)
    model = _build_model(spec).to(device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    effective_beta = spec.model.fixed_beta if args.beta is None else args.beta
    if (
        not spec.model.condition_on_beta
        and abs(effective_beta - spec.model.fixed_beta) > 1e-12
    ):
        raise ValueError("A fixed-beta checkpoint cannot sample a different beta")

    result = sample_absorbing(
        model,
        (args.batch_size, args.width, args.width),
        beta=effective_beta,
        steps=args.steps,
        sampling_temperature=args.temperature,
        seed=args.seed,
        device=device,
    )
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else spec.output_dir / "samples" / f"W{args.width}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"samples_seed{args.seed}"
    sample_path = output_dir / f"{stem}.npz"
    np.savez_compressed(sample_path, spins=result.spins.cpu().numpy().astype(np.int8))
    atomic_write_text(
        output_dir / f"{stem}_trace.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result.trace),
    )
    atomic_write_json(
        output_dir / f"{stem}_manifest.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config": str(spec.source_path),
            "width": args.width,
            "batch_size": args.batch_size,
            "beta": effective_beta,
            "condition_on_beta": spec.model.condition_on_beta,
            "reverse_steps": args.steps,
            "schedule": "cosine",
            "sampling_temperature": args.temperature,
            "seed": args.seed,
            "observables": summarize(result.spins.cpu(), min(16, args.width - 1)),
        },
    )
    print(sample_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
