from __future__ import annotations

from pathlib import Path

from ising_scale_diffusion.spec import BETA_CRITICAL, load_experiment


def test_shipped_configs_fix_beta_but_keep_optional_interface() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("train_smoke.yaml", "train_critical.yaml"):
        spec = load_experiment(root / "configs" / name)
        assert spec.model.condition_on_beta is False
        assert spec.model.fixed_beta == BETA_CRITICAL
        assert spec.training.adam_betas == (0.9, 0.95)
