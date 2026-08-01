import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import ViTConfig, VisionTransformer
from src.radial_audit import run_radial_audit


def test_radial_restore(tmp_path):
    cfg = ViTConfig(embed_dim=24, depth=1, num_heads=3, num_classes=4)
    model = VisionTransformer(cfg)
    state = copy.deepcopy(model.state_dict())
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])), batch_size=2)
    run_radial_audit(model, state, loader, "cpu", [0.5], tmp_path, generate_plot=False)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_radial_restore_from_cpu_state_on_cuda(tmp_path):
    cfg = ViTConfig(embed_dim=24, depth=1, num_heads=3, num_classes=4)
    model = VisionTransformer(cfg).cuda()
    state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])), batch_size=2)
    run_radial_audit(model, state, loader, "cuda", [0.5], tmp_path,
                     generate_plot=False)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in state.items())
