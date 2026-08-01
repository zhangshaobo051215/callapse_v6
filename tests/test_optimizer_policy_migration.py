import copy

import torch
from torch.nn import functional as F

from src.model import ViTConfig, VisionTransformer
from src.optimizer_migration import rebuild_optimizer_with_policy
from src.param_groups import build_optimizer


CFG = ViTConfig(
    image_size=16, embed_dim=12, depth=1, num_heads=3, num_classes=4)


def _step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()


def test_optimizer_policy_migration_preserves_adam_state_and_next_update():
    torch.manual_seed(17)
    source_model = VisionTransformer(CFG)
    source_optimizer, _, _ = build_optimizer(
        source_model, "hidden_matrices", 1e-3, weight_decay=0)
    for _ in range(3):
        _step(
            source_model,
            source_optimizer,
            torch.randn(2, 3, 16, 16),
            torch.randint(0, 4, (2,)),
        )
    model_state = copy.deepcopy(source_model.state_dict())
    optimizer_state = copy.deepcopy(source_optimizer.state_dict())

    reference_model = VisionTransformer(CFG)
    reference_model.load_state_dict(model_state)
    reference_optimizer, _, _ = build_optimizer(
        reference_model, "hidden_matrices", 1e-3, weight_decay=0)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer_state))

    migrated_model = VisionTransformer(CFG)
    migrated_model.load_state_dict(model_state)
    migrated_optimizer, controlled, uncontrolled = rebuild_optimizer_with_policy(
        migrated_model,
        copy.deepcopy(optimizer_state),
        "hidden_matrices",
        "all_2d",
        1e-3,
        weight_decay=0,
    )
    assert "head.weight" in controlled
    assert "head.bias" in uncontrolled

    reference_params = dict(reference_model.named_parameters())
    migrated_params = dict(migrated_model.named_parameters())
    for name in reference_params:
        reference_state = reference_optimizer.state[reference_params[name]]
        migrated_state = migrated_optimizer.state[migrated_params[name]]
        assert set(reference_state) == set(migrated_state)
        for key in reference_state:
            assert torch.equal(reference_state[key], migrated_state[key])

    x = torch.randn(2, 3, 16, 16)
    y = torch.randint(0, 4, (2,))
    _step(reference_model, reference_optimizer, x, y)
    _step(migrated_model, migrated_optimizer, x, y)
    for name in reference_params:
        assert torch.equal(
            reference_model.state_dict()[name],
            migrated_model.state_dict()[name],
        )
