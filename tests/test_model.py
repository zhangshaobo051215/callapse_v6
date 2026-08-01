import torch

from src.model import ViTConfig, VisionTransformer


def test_model_shape_names_and_geometry():
    model = VisionTransformer(ViTConfig())
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 200)
    assert model.patch_embed.num_patches == 64
    assert model.blocks[0].attn.head_dim == 64
    names = dict(model.named_parameters())
    assert "blocks.0.attn.qkv.weight" in names
    assert "blocks.11.mlp.fc2.weight" in names

