from __future__ import annotations

import csv
import io
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .artifacts import JsonlLogger, atomic_torch_save, atomic_write_text, sha256_file
from .data import FixedCropSource, ReplayableCropSource
from .evaluator import evaluate_fixed_nelbo
from .spec import ExperimentSpec
from .system import IsingDiffusionSystem


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _learning_rate_scale(step: int, spec: ExperimentSpec) -> float:
    warmup = spec.training.warmup_steps
    total = spec.training.steps
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    denominator = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / denominator))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    minimum = spec.training.min_lr_ratio
    return minimum + (1.0 - minimum) * cosine


class Trainer:
    def __init__(
        self,
        spec: ExperimentSpec,
        system: IsingDiffusionSystem,
        train_source: ReplayableCropSource,
        validation_source: FixedCropSource | ReplayableCropSource,
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.system = system.to(device)
        self.train_source = train_source
        self.validation_source = validation_source
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.system.parameters(),
            lr=spec.training.learning_rate,
            betas=spec.training.adam_betas,
            weight_decay=spec.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: _learning_rate_scale(step, spec)
        )
        scaler_enabled = device.type == "cuda" and spec.training.precision == "fp16"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:  # PyTorch releases before the device-agnostic AMP API.
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.logger = JsonlLogger(spec.output_dir / "logs" / "train.jsonl")
        self.global_step = 0
        self.processed_sites = 0
        self.best_validation = math.inf
        self.started_at = time.time()

    def _autocast(self):
        precision = self.spec.training.precision
        if precision == "fp32" or self.device.type not in {"cuda", "cpu"}:
            return nullcontext()
        if self.device.type == "cpu" and precision == "fp16":
            return nullcontext()
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            "model": self.system.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "processed_sites": self.processed_sites,
            "best_validation": self.best_validation,
            "wall_time_seconds": time.time() - self.started_at,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "config_sha256": sha256_file(self.spec.source_path),
            "model_facts": self.system.model.architecture_facts(),
            "data_state": {"next_step": self.global_step},
        }

    def save_checkpoint(self, name: str) -> Path:
        destination = self.spec.output_dir / "checkpoints" / name
        atomic_torch_save(destination, self._checkpoint_state())
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        source = Path(path).expanduser().resolve()
        try:
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(source, map_location="cpu")
        expected_hash = sha256_file(self.spec.source_path)
        if checkpoint.get("config_sha256") != expected_hash:
            raise ValueError("Checkpoint config hash does not match the requested run")
        self.system.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.global_step = int(checkpoint["global_step"])
        self.processed_sites = int(checkpoint["processed_sites"])
        self.best_validation = float(checkpoint.get("best_validation", math.inf))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])

    @torch.no_grad()
    def validate(self) -> float:
        result = evaluate_fixed_nelbo(
            self.system,
            self.validation_source,
            self.spec.data.widths,
            self.spec.validation.t_grid,
            self.spec.validation.batches_per_width,
            self.device,
            self._autocast,
            self.global_step,
        )
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(result.rows[0]))
        writer.writeheader()
        writer.writerows(result.rows)
        atomic_write_text(
            self.spec.output_dir / "validation" / "nelbo_by_t.csv", buffer.getvalue()
        )
        self.logger.write(
            {
                "kind": "validation",
                "step": self.global_step,
                "val/nelbo": result.nelbo,
                "widths": list(self.spec.data.widths),
            }
        )
        if result.nelbo < self.best_validation:
            self.best_validation = result.nelbo
            self.save_checkpoint("best.pt")
        return result.nelbo

    def fit(self) -> None:
        accumulation = self.spec.training.gradient_accumulation
        self.system.train()
        for step in range(self.global_step, self.spec.training.steps):
            step_started = time.time()
            width = self.train_source.width_for_step(step)
            self.optimizer.zero_grad(set_to_none=True)
            aggregate = {
                "loss": 0.0,
                "masked_ce": 0.0,
                "masked_accuracy": 0.0,
                "mask_fraction": 0.0,
                "diffusion_t": 0.0,
            }
            step_sites = 0
            for microbatch in range(accumulation):
                batch = self.train_source.batch(
                    step,
                    microbatch=microbatch,
                    accumulation_steps=accumulation,
                    width=width,
                )
                step_sites += int(batch["n_sites"])
                batch = _move_batch(batch, self.device)
                with self._autocast():
                    result = self.system.loss_with_metrics(batch)
                    scaled_loss = result.loss / accumulation
                self.scaler.scale(scaled_loss).backward()
                aggregate["loss"] += float(result.loss.detach().item()) / accumulation
                aggregate["masked_ce"] += (
                    float(result.masked_ce.detach().item()) / accumulation
                )
                aggregate["masked_accuracy"] += (
                    float(result.masked_accuracy.detach().item()) / accumulation
                )
                aggregate["mask_fraction"] += (
                    float(result.mask_fraction.detach().item()) / accumulation
                )
                aggregate["diffusion_t"] += (
                    float(result.diffusion_t_mean.detach().item()) / accumulation
                )

            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.system.parameters(), self.spec.training.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step = step + 1
            self.processed_sites += step_sites

            if self.global_step % self.spec.training.log_every == 0 or step == 0:
                peak_memory = (
                    torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == "cuda"
                    else 0
                )
                row = {
                    "kind": "train",
                    "step": self.global_step,
                    "width": width,
                    "loss": aggregate["loss"],
                    "masked_ce": aggregate["masked_ce"],
                    "masked_accuracy": aggregate["masked_accuracy"],
                    "mask_fraction": aggregate["mask_fraction"],
                    "diffusion_t": aggregate["diffusion_t"],
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "processed_sites": self.processed_sites,
                    "step_seconds": time.time() - step_started,
                    "peak_cuda_bytes": peak_memory,
                }
                self.logger.write(row)
                print(
                    f"step={self.global_step} width={width} "
                    f"loss={aggregate['loss']:.6f} "
                    f"acc={aggregate['masked_accuracy']:.4f}",
                    flush=True,
                )

            if self.global_step % self.spec.training.validate_every == 0:
                value = self.validate()
                print(
                    f"validation step={self.global_step} nelbo={value:.6f}", flush=True
                )
            if self.global_step % self.spec.training.checkpoint_every == 0:
                self.save_checkpoint(f"step_{self.global_step:08d}.pt")
                self.save_checkpoint("last.pt")

        if self.global_step % self.spec.training.validate_every:
            self.validate()
        self.save_checkpoint("last.pt")
