from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ViTConfig:
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 3
    num_classes: int = 200
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    norm_eps: float = 1e-6
    norm_type: str = "layernorm"
    init_std: float = 0.02


class RMSNorm(nn.Module):
    """Bias-free RMS normalization over the embedding dimension."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(
            x.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return x * inverse_rms * self.weight


def build_norm(cfg: ViTConfig) -> nn.Module:
    norm_type = cfg.norm_type.lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps)
    if norm_type == "rmsnorm":
        return RMSNorm(cfg.embed_dim, eps=cfg.norm_eps)
    raise ValueError(f"unsupported norm_type: {cfg.norm_type}")


class PatchEmbed(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        if cfg.image_size % cfg.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.proj = nn.Conv2d(cfg.in_channels, cfg.embed_dim, cfg.patch_size, cfg.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        if cfg.embed_dim % cfg.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=cfg.qkv_bias)
        self.attn_drop = nn.Dropout(cfg.attn_drop_rate)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.proj_drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        if cfg.drop_path_rate != 0:
            raise ValueError("v1.0 requires drop_path_rate=0")
        self.norm1 = build_norm(cfg)
        self.attn = Attention(cfg)
        self.norm2 = build_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig | dict):
        super().__init__()
        cfg = ViTConfig(**cfg) if isinstance(cfg, dict) else cfg
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.drop_rate)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = build_norm(cfg)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=cfg.init_std)
        nn.init.trunc_normal_(self.pos_embed, std=cfg.init_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.pos_drop(torch.cat((cls, x), dim=1) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

