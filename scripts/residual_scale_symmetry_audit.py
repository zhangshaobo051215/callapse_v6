from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import VisionTransformer  # noqa: E402


SCALES = (1 / 3, 0.5, 1.5, 2.0)
CASE_ORDER = (
    "raw_pos_head",
    "no_head",
    "input_family",
    "residual_stream_symmetry",
)
_HIDDEN_WEIGHT_SUFFIXES = (
    ".attn.qkv.weight",
    ".attn.proj.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
)
_RESIDUAL_OUTPUT_SUFFIXES = (
    ".attn.proj.weight",
    ".attn.proj.bias",
    ".mlp.fc2.weight",
    ".mlp.fc2.bias",
)
_INPUT_FAMILY = {
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "cls_token",
    "pos_embed",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state(model: VisionTransformer) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def transformation_parameter_names(
        model: VisionTransformer) -> dict[str, tuple[str, ...]]:
    """Return the exact physical parameters scaled by each diagnostic."""
    named = dict(model.named_parameters())
    hidden = {
        name for name in named
        if name.endswith(_HIDDEN_WEIGHT_SUFFIXES)
    }
    raw_pos_head = hidden | {
        "patch_embed.proj.weight",
        "pos_embed",
        "head.weight",
    }
    no_head = raw_pos_head - {"head.weight"}
    residual_stream = _INPUT_FAMILY | {
        name for name in named
        if name.endswith(_RESIDUAL_OUTPUT_SUFFIXES)
    }
    requested = {
        "raw_pos_head": raw_pos_head,
        "no_head": no_head,
        "input_family": set(_INPUT_FAMILY),
        "residual_stream_symmetry": residual_stream,
    }
    result = {}
    for case in CASE_ORDER:
        missing = sorted(requested[case] - set(named))
        if missing:
            raise ValueError(
                f"{case} requires model parameters that are missing: {missing}")
        result[case] = tuple(
            name for name in named if name in requested[case])
    return result


@torch.no_grad()
def apply_transformation(
        model: VisionTransformer, case: str, scale: float) -> tuple[str, ...]:
    """Scale one case in-place and return its selected parameter names."""
    if case not in CASE_ORDER:
        raise ValueError(f"unknown symmetry audit case: {case}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    groups = transformation_parameter_names(model)
    named = dict(model.named_parameters())
    for name in groups[case]:
        named[name].mul_(scale)
    return groups[case]


class _ActivationRecorder:
    def __init__(self, model: VisionTransformer):
        self._input_sum_sq = 0.0
        self._input_numel = 0
        self._block_sum_sq = [0.0 for _ in model.blocks]
        self._block_numel = [0 for _ in model.blocks]
        self.features: list[torch.Tensor] = []
        self._handles = [
            model.blocks[0].register_forward_pre_hook(self._input_hook),
            model.norm.register_forward_hook(self._feature_hook),
        ]
        self._handles.extend(
            block.register_forward_hook(self._block_hook(index))
            for index, block in enumerate(model.blocks)
        )

    def _input_hook(self, _module, args) -> None:
        value = args[0].detach()
        self._input_sum_sq += value.double().square().sum().item()
        self._input_numel += value.numel()

    def _block_hook(self, index):
        def hook(_module, _args, output) -> None:
            value = output.detach()
            self._block_sum_sq[index] += (
                value.double().square().sum().item())
            self._block_numel[index] += value.numel()
        return hook

    def _feature_hook(self, _module, _args, output) -> None:
        self.features.append(output[:, 0].detach().cpu())

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def summary(self) -> tuple[float, list[float], torch.Tensor]:
        if self._input_numel == 0 or any(
                count == 0 for count in self._block_numel):
            raise RuntimeError("activation hooks did not observe a complete forward")
        if not self.features:
            raise RuntimeError("final LayerNorm hook did not observe features")
        input_rms = math.sqrt(self._input_sum_sq / self._input_numel)
        block_rms = [
            math.sqrt(total / count)
            for total, count in zip(
                self._block_sum_sq, self._block_numel)
        ]
        return input_rms, block_rms, torch.cat(self.features)


@torch.inference_mode()
def _evaluate(
        model: VisionTransformer,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        device: torch.device,
) -> dict:
    model.eval()
    recorder = _ActivationRecorder(model)
    logits_parts = []
    loss_sum = 0.0
    try:
        for start in range(0, images.shape[0], batch_size):
            stop = min(start + batch_size, images.shape[0])
            batch_images = images[start:stop].to(device)
            batch_labels = labels[start:stop].to(device)
            logits = model(batch_images)
            loss_sum += F.cross_entropy(
                logits, batch_labels, reduction="sum").item()
            logits_parts.append(logits.detach().cpu())
    finally:
        recorder.close()
    input_rms, block_rms, features = recorder.summary()
    logits = torch.cat(logits_parts)
    return {
        "ce_loss": loss_sum / labels.numel(),
        "logits": logits,
        "features": features,
        "logits_rms": _rms(logits),
        "features_rms": _rms(features),
        "input_tokens_rms": input_rms,
        "block_residual_rms": block_rms,
    }


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().double().square().mean().sqrt().item())


def _difference(
        current: torch.Tensor, baseline: torch.Tensor,
        eps: float = 1e-12) -> dict[str, float]:
    current = current.detach().double()
    baseline = baseline.detach().double()
    if current.shape != baseline.shape:
        raise ValueError(
            f"comparison shape mismatch: {current.shape} != {baseline.shape}")
    delta = current - baseline
    max_abs = float(delta.abs().max().item())
    rms = float(delta.square().mean().sqrt().item())
    baseline_max = float(baseline.abs().max().item())
    baseline_rms = float(baseline.square().mean().sqrt().item())
    return {
        "max_abs_diff": max_abs,
        "max_abs_relative": max_abs / max(baseline_max, eps),
        "rms_diff": rms,
        "rms_relative": rms / max(baseline_rms, eps),
    }


def _public_evaluation(evaluation: dict) -> dict:
    return {
        "ce_loss": evaluation["ce_loss"],
        "logits_rms": evaluation["logits_rms"],
        "features_rms": evaluation["features_rms"],
        "input_tokens_rms": evaluation["input_tokens_rms"],
        "block_residual_rms": evaluation["block_residual_rms"],
    }


def _transformation_errors(
        model: VisionTransformer,
        baseline_state: Mapping[str, torch.Tensor],
        selected: Sequence[str],
        scale: float,
) -> dict[str, float]:
    selected_set = set(selected)
    current = model.state_dict()
    scaled_error = 0.0
    unscaled_error = 0.0
    for name, reference in baseline_state.items():
        actual = current[name].detach().cpu()
        expected = reference * scale if name in selected_set else reference
        error = float((actual - expected).abs().max().item())
        if name in selected_set:
            scaled_error = max(scaled_error, error)
        else:
            unscaled_error = max(unscaled_error, error)
    return {
        "scaled_parameter_max_abs_error": scaled_error,
        "unscaled_state_max_abs_error": unscaled_error,
    }


def _result_row(
        case: str,
        scale: float,
        selected: Sequence[str],
        model: VisionTransformer,
        baseline_state: Mapping[str, torch.Tensor],
        baseline: dict,
        current: dict,
) -> dict:
    block_rows = []
    for index, (base_rms, current_rms) in enumerate(zip(
            baseline["block_residual_rms"],
            current["block_residual_rms"])):
        ratio = current_rms / max(base_rms, 1e-12)
        block_rows.append({
            "block": index,
            "baseline_rms": base_rms,
            "scaled_rms": current_rms,
            "rms_ratio": ratio,
            "ratio_over_q": ratio / scale,
            "relative_error_to_q": abs(ratio - scale) / scale,
        })
    input_ratio = (
        current["input_tokens_rms"]
        / max(baseline["input_tokens_rms"], 1e-12)
    )
    named = dict(model.named_parameters())
    return {
        "case": case,
        "q": scale,
        "scaled_parameters": list(selected),
        "scaled_parameter_count": len(selected),
        "scaled_numel": sum(named[name].numel() for name in selected),
        "transformation_check": _transformation_errors(
            model, baseline_state, selected, scale),
        "ce_loss": current["ce_loss"],
        "ce_loss_diff": current["ce_loss"] - baseline["ce_loss"],
        "absolute_ce_loss_diff": abs(
            current["ce_loss"] - baseline["ce_loss"]),
        "logits_rms": current["logits_rms"],
        "logits_rms_ratio": (
            current["logits_rms"]
            / max(baseline["logits_rms"], 1e-12)),
        "features_rms": current["features_rms"],
        "features_rms_ratio": (
            current["features_rms"]
            / max(baseline["features_rms"], 1e-12)),
        "logits_difference": _difference(
            current["logits"], baseline["logits"]),
        "features_difference": _difference(
            current["features"], baseline["features"]),
        "input_tokens": {
            "baseline_rms": baseline["input_tokens_rms"],
            "scaled_rms": current["input_tokens_rms"],
            "rms_ratio": input_ratio,
            "ratio_over_q": input_ratio / scale,
            "relative_error_to_q": abs(input_ratio - scale) / scale,
        },
        "block_residuals": block_rows,
        "first_block_residual": block_rows[0],
        "last_block_residual": block_rows[-1],
        "max_block_relative_error_to_q": max(
            row["relative_error_to_q"] for row in block_rows),
    }


def _load_checkpoint(
        path: Path) -> tuple[dict, dict, Mapping[str, torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint has no model state mapping")
    model_config = checkpoint.get("config", {}).get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint has no config.model mapping")
    return dict(checkpoint), dict(model_config), model_state


def run_audit(
        checkpoint_path: Path | str,
        *,
        device: str | torch.device = "cpu",
        seed: int = 20260729,
        probe_size: int = 32,
        batch_size: int = 8,
        scales: Sequence[float] = SCALES,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint, model_config, model_state = _load_checkpoint(checkpoint_path)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if probe_size <= 0 or batch_size <= 0:
        raise ValueError("probe_size and batch_size must be positive")
    scales = tuple(float(scale) for scale in scales)
    if not scales or any(
            not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("all scales must be finite and positive")

    model = VisionTransformer(model_config).to(device)
    model.load_state_dict(model_state, strict=True)
    baseline_state = _cpu_state(model)
    groups = transformation_parameter_names(model)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image_shape = (
        probe_size,
        int(model_config["in_channels"]),
        int(model_config["image_size"]),
        int(model_config["image_size"]),
    )
    images = torch.randn(image_shape, generator=generator)
    labels = torch.randint(
        0, int(model_config["num_classes"]),
        (probe_size,), generator=generator)

    model.load_state_dict(baseline_state, strict=True)
    baseline = _evaluate(model, images, labels, batch_size, device)
    rows = []
    for case in CASE_ORDER:
        for scale in scales:
            model.load_state_dict(baseline_state, strict=True)
            selected = apply_transformation(model, case, scale)
            current = _evaluate(
                model, images, labels, batch_size, device)
            rows.append(_result_row(
                case,
                scale,
                selected,
                model,
                baseline_state,
                baseline,
                current,
            ))

    model.load_state_dict(baseline_state, strict=True)
    restored = _cpu_state(model)
    mismatches = [
        name for name in baseline_state
        if not torch.equal(restored[name], baseline_state[name])
    ]
    if mismatches:
        raise AssertionError(
            f"audit failed to restore baseline state: {mismatches[:5]}")

    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "global_state_step": checkpoint.get("global_state_step"),
        "device": str(device),
        "scales": list(scales),
        "synthetic_probe": {
            "seed": seed,
            "probe_size": probe_size,
            "batch_size": batch_size,
            "image_shape": list(image_shape[1:]),
            "label_histogram": torch.bincount(
                labels, minlength=int(model_config["num_classes"])).tolist(),
        },
        "transformations": {
            case: {
                "scaled_parameters": list(groups[case]),
                "scaled_parameter_count": len(groups[case]),
            }
            for case in CASE_ORDER
        },
        "baseline": _public_evaluation(baseline),
        "results": rows,
        "restoration_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-training forward audit of ViT residual-scale "
            "symmetry candidates."))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--probe-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    result = run_audit(
        args.checkpoint,
        device=args.device,
        seed=args.seed,
        probe_size=args.probe_size,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "device": result["device"],
        "probe_size": result["synthetic_probe"]["probe_size"],
        "cases": len(CASE_ORDER),
        "scales": result["scales"],
        "output": str(args.output.resolve()),
        "restoration_verified": result["restoration_verified"],
    }, indent=2))


if __name__ == "__main__":
    main()
