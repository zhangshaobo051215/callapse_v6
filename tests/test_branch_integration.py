import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import EpochBatchSampler
from src.model import ViTConfig, VisionTransformer
from src.norm_control import reference_norms_for
from src.param_groups import build_optimizer
from src.train_branch import train_branch


def test_two_step_controlled_branch(tmp_path):
    torch.manual_seed(1)
    model = VisionTransformer(ViTConfig(
        image_size=16, embed_dim=12, depth=1, num_heads=3, num_classes=4))
    optimizer, controlled, _ = build_optimizer(model, "hidden_matrices", 1e-4)
    refs = reference_norms_for(controlled, split_fused_qkv=True)
    dataset = TensorDataset(torch.randn(8, 3, 16, 16), torch.randint(0, 4, (8,)))
    sampler = EpochBatchSampler(len(dataset), 2, 11)
    factory = lambda s: DataLoader(dataset, batch_sampler=s)
    cfg = {
        "data": {"global_batch_size": 2, "micro_batch_size": 2},
        "optimizer": {"peak_lr": 1e-4, "final_lr": 1e-5, "warmup_steps": 1,
                      "control_start_step": 0, "decay_start_step": 1, "total_steps": 2},
        "control": {"cyclic_period_steps": 2, "cyclic_amplitude": .5,
                    "linear_up_final": 2., "linear_down_final": 1 / 3,
                    "projection_eps": 1e-12,
                    "split_fused_qkv": True},
        "logging": {"ema_beta": .99, "angular_interval": 1, "probe_interval": 100,
                    "val_interval": 100, "checkpoint_interval": 2},
    }
    train_branch(cfg, "linear_up", model, optimizer, controlled, sampler, factory, "cpu",
                 tmp_path, 0, None, refs)
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert len(metrics) == 2
    assert metrics.target_norm_relative_error_max.max() < 1e-5
    assert metrics.elr_relative_error_max.max() < 1e-5
    assert (tmp_path / "checkpoint_step_000002.pt").exists()
