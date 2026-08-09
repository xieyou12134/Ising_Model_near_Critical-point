from __future__ import annotations

import torch
from torch import nn

from .model import IsingDiffusionModel
from .objective import AbsorbingDiffusionObjective, LossResult


class IsingDiffusionSystem(nn.Module):
    """Owns corruption and NELBO while the model owns only clean-spin logits."""

    def __init__(
        self,
        model: IsingDiffusionModel,
        objective: AbsorbingDiffusionObjective,
        fixed_beta_tolerance: float = 1e-7,
    ) -> None:
        super().__init__()
        self.model = model
        self.objective = objective
        self.fixed_beta_tolerance = float(fixed_beta_tolerance)

    def _check_fixed_beta(self, beta: torch.Tensor) -> None:
        if self.model.condition_on_beta:
            return
        expected = torch.full_like(beta.float(), self.model.fixed_beta)
        if not torch.allclose(
            beta.float(), expected, rtol=0.0, atol=self.fixed_beta_tolerance
        ):
            raise ValueError(
                "This run fixes beta at beta_c; the batch contains a different beta"
            )

    def loss_with_metrics(
        self,
        batch: dict[str, torch.Tensor],
        diffusion_t: torch.Tensor | float | None = None,
    ) -> LossResult:
        clean_tokens = batch["clean_tokens"]
        beta = batch["beta"]
        self._check_fixed_beta(beta)
        corruption = self.objective.corrupt(
            clean_tokens, batch["noise_seeds"], diffusion_t=diffusion_t
        )
        model_beta = beta if self.model.condition_on_beta else None
        logits = self.model(
            corruption.noisy_tokens, corruption.diffusion_t, beta=model_beta
        )
        return self.objective.loss_from_logits(
            logits, clean_tokens, corruption.mask, corruption.diffusion_t
        )

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss_with_metrics(batch).loss
