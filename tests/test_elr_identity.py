import pytest

from src.norm_control import rms_from_norm


@pytest.mark.parametrize("q", [1, 2, 1 / 3, .5, 1.5])
def test_elr_identity(q):
    lr, norm, numel = 6e-4, 11.2, 192 * 192
    assert q * lr / rms_from_norm(q * norm, numel) == pytest.approx(
        lr / rms_from_norm(norm, numel))

