import pytest
import torch

from src.norm_control import project_frobenius_


def test_projection_norm_and_direction():
    x = torch.randn(7, 5)
    old = x.clone()
    project_frobenius_(x, 3.25)
    assert torch.linalg.vector_norm(x).item() == pytest.approx(3.25, rel=1e-6)
    cosine = torch.nn.functional.cosine_similarity(old.flatten(), x.flatten(), dim=0)
    assert cosine.item() == pytest.approx(1, abs=1e-6)


def test_zero_projection_fails():
    with pytest.raises(ValueError):
        project_frobenius_(torch.zeros(2, 2), 1)

