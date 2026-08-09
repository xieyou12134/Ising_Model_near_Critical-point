from __future__ import annotations

import pytest
import torch

from ising_scale_diffusion.model import IsingDiffusionModel, SharedAxialAttention


def test_model_accepts_rectangular_fields_and_optional_beta() -> None:
    model = IsingDiffusionModel(
        dim=32, depth=2, heads=4, condition_on_beta=False, dropout=0.0
    )
    tokens = torch.randint(0, 3, (2, 6, 10))
    diffusion_t = torch.tensor([0.4, 0.8])
    logits = model(tokens, diffusion_t)
    assert logits.shape == (2, 6, 10, 2)
    assert model.architecture_facts()["learned_absolute_position"] is False


def test_beta_is_required_only_when_conditioning_is_enabled() -> None:
    model = IsingDiffusionModel(dim=32, depth=1, heads=4, condition_on_beta=True)
    with pytest.raises(ValueError, match="beta is required"):
        model(torch.zeros((1, 4, 4), dtype=torch.long), torch.tensor([0.5]))
    output = model(
        torch.zeros((1, 4, 4), dtype=torch.long),
        torch.tensor([0.5]),
        beta=torch.tensor([0.4406868]),
    )
    assert output.shape == (1, 4, 4, 2)


def test_shared_axial_attention_is_transpose_equivariant() -> None:
    torch.manual_seed(4)
    attention = SharedAxialAttention(32, 4, dropout=0.0).eval()
    field = torch.randn(2, 5, 7, 32)
    direct = attention(field)
    transposed = attention(field.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(direct, transposed, rtol=1e-5, atol=1e-6)
