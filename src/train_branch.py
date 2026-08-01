from __future__ import annotations

import math
import statistics
from pathlib import Path

import torch

from .checkpointing import save_checkpoint
from .data import infinite_batches
from .norm_control import (angular_steps, control_units, frobenius_norm,
                           project_controlled, rms_from_norm)
from .radial_audit import evaluate
from .schedules import base_lr_at, schedule_ratio
from .train_prefix import optimizer_step
from .utils import (append_csv, append_csv_rows, last_csv_values,
                    truncate_csv_after_step)


def _set_lrs(optimizer, base_lr, ratio):
    for group in optimizer.param_groups:
        group["lr"] = base_lr * ratio if group.get("name") == "controlled" else base_lr


def _optimizer_assignments(model, optimizer):
    trainable = {name: param for name, param in model.named_parameters()
                 if param.requires_grad}
    names_by_id = {id(param): name for name, param in trainable.items()}
    assignments = {}
    for index, group in enumerate(optimizer.param_groups):
        label = str(group.get("name", f"group_{index}"))
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter outside model trainable tensors")
            if name in assignments:
                raise ValueError(f"optimizer contains duplicate parameter {name}")
            assignments[name] = (label, group)
    missing = set(trainable) - set(assignments)
    if missing:
        raise ValueError(f"optimizer is missing trainable tensors: {sorted(missing)}")
    return trainable, assignments


def _batched_frobenius_norms(parameters):
    """Measure many tensors with one device-to-host synchronization."""
    names = list(parameters)
    if not names:
        return {}
    with torch.no_grad():
        values = torch.stack([
            torch.linalg.vector_norm(parameters[name].detach().float())
            for name in names
        ]).cpu().tolist()
    return dict(zip(names, values))


def _validate_monitor_references(trainable, references):
    expected, actual = set(trainable), set(references)
    if actual != expected:
        raise ValueError(
            "tensor monitor reference keys must exactly match trainable tensors; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    result = {name: float(references[name]) for name in trainable}
    invalid = [name for name, value in result.items()
               if not math.isfinite(value) or value < 0]
    if invalid:
        raise ValueError(f"tensor monitor references must be finite and nonnegative: {invalid}")
    return result


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator != 0 else math.nan


def _tensor_monitor_rows(*, branch, step, base_lr, schedule_ratio_now,
                         schedule_ratio_next, trainable, controlled,
                         assignments, references, pre_norms,
                         post_optimizer_norms, post_projection_norms):
    rows = []
    controlled_names = set(controlled)
    for name, parameter in trainable.items():
        is_controlled = name in controlled_names
        optimizer_group, group = assignments[name]
        actual_lr = float(group["lr"])
        reference = references[name]
        pre = pre_norms[name]
        post_optimizer = post_optimizer_norms[name]
        post_projection = post_projection_norms[name]
        rms_scale = math.sqrt(parameter.numel())
        lr_over_frobenius = _safe_ratio(actual_lr, pre)
        lr_over_rms = _safe_ratio(actual_lr, pre / rms_scale)
        reference_lr_over_frobenius = _safe_ratio(base_lr, reference)
        reference_lr_over_rms = _safe_ratio(base_lr, reference / rms_scale)
        target_post = schedule_ratio_next * reference if is_controlled else math.nan
        projection_error = (
            _safe_ratio(abs(post_projection - target_post), abs(target_post))
            if is_controlled else math.nan)
        rows.append({
            "update_start_state_step": step,
            "global_state_step": step + 1,
            "branch": branch,
            "tensor": name,
            "optimizer_group": optimizer_group,
            "is_controlled": is_controlled,
            "ndim": parameter.ndim,
            "numel": parameter.numel(),
            "base_lr": base_lr,
            "actual_lr": actual_lr,
            "schedule_ratio": schedule_ratio_now,
            "next_schedule_ratio": schedule_ratio_next,
            "effective_schedule_ratio": schedule_ratio_now if is_controlled else 1.0,
            "reference_frobenius_norm": reference,
            "pre_update_frobenius_norm": pre,
            "post_optimizer_pre_projection_frobenius_norm": post_optimizer,
            "post_projection_frobenius_norm": post_projection,
            "pre_norm_ratio_to_reference": _safe_ratio(pre, reference),
            "post_optimizer_norm_ratio_to_reference": _safe_ratio(
                post_optimizer, reference),
            "post_projection_norm_ratio_to_reference": _safe_ratio(
                post_projection, reference),
            "lr_over_frobenius_norm": lr_over_frobenius,
            "lr_over_rms": lr_over_rms,
            "reference_lr_over_frobenius_norm": reference_lr_over_frobenius,
            "reference_lr_over_rms": reference_lr_over_rms,
            "lr_over_frobenius_ratio_to_reference": _safe_ratio(
                lr_over_frobenius, reference_lr_over_frobenius),
            "lr_over_rms_ratio_to_reference": _safe_ratio(
                lr_over_rms, reference_lr_over_rms),
            "target_post_frobenius_norm": target_post,
            "post_projection_relative_error": projection_error,
        })
    return rows


def train_branch(cfg, branch, model, optimizer, controlled, sampler, loader_factory, device,
                 output, start_step, ema, reference_norms, probe_loader=None, val_loader=None,
                 monitor_reference_norms=None,
                 control_unit_monitor_reference_norms=None):
    output, oc, cc, lc = Path(output), cfg["optimizer"], cfg["control"], cfg["logging"]
    accumulation = cfg["data"]["global_batch_size"] // cfg["data"]["micro_batch_size"]
    batches = infinite_batches(loader_factory, sampler)
    split_qkv = bool(cc.get("split_fused_qkv", False))
    units = control_units(controlled, split_fused_qkv=split_qkv)
    metrics_path = output / "metrics.csv"
    evaluation_path = output / "evaluation_metrics.csv"
    angular_path = output / "angular_metrics.csv"
    monitor_path = output / "tensor_metrics.csv"
    unit_monitor_path = output / "control_unit_metrics.csv"
    truncate_csv_after_step(
        metrics_path, start_step,
        dedupe_by=("branch", "global_state_step"))
    truncate_csv_after_step(
        evaluation_path, start_step,
        dedupe_by=("branch", "split", "global_state_step"))
    truncate_csv_after_step(
        angular_path, start_step,
        dedupe_by=("branch", "tensor", "global_state_step"))
    cumulative = {n: 0.0 for n in controlled}
    cumulative.update({
        name: value
        for name, value in last_csv_values(
            angular_path, "tensor", "cumulative_angular_step",
            branch=branch,
        ).items()
        if name in cumulative
    })
    monitor_enabled = monitor_reference_norms is not None
    unit_monitor_enabled = control_unit_monitor_reference_norms is not None
    if unit_monitor_enabled and not monitor_enabled:
        raise ValueError("control-unit monitoring requires named-tensor monitoring")
    if monitor_enabled:
        trainable, assignments = _optimizer_assignments(model, optimizer)
        if any(name not in trainable or trainable[name] is not parameter
               for name, parameter in controlled.items()):
            raise ValueError("controlled mapping does not match model trainable tensors")
        monitor_reference_norms = _validate_monitor_references(
            trainable, monitor_reference_norms)
        tensor_interval = int(lc.get(
            "tensor_monitor_interval", lc["angular_interval"]))
        if tensor_interval <= 0:
            raise ValueError("logging.tensor_monitor_interval must be positive")
        truncate_csv_after_step(
            monitor_path, start_step,
            dedupe_by=("branch", "tensor", "global_state_step"))
    if unit_monitor_enabled:
        control_unit_monitor_reference_norms = _validate_monitor_references(
            units, control_unit_monitor_reference_norms)
        unit_assignments = {}
        for name in units:
            parent = name.partition("::")[0]
            if parent not in assignments:
                raise ValueError(f"control unit has no optimizer parent: {name}")
            unit_assignments[name] = assignments[parent]
        truncate_csv_after_step(
            unit_monitor_path, start_step,
            dedupe_by=("branch", "tensor", "global_state_step"))
    for step in range(start_step, oc["total_steps"]):
        kwargs = dict(control_start_step=oc["control_start_step"], total_steps=oc["total_steps"],
                      cyclic_period_steps=cc["cyclic_period_steps"],
                      cyclic_amplitude=cc["cyclic_amplitude"],
                      linear_up_final=cc["linear_up_final"],
                      linear_down_final=cc["linear_down_final"])
        q_now, q_next = schedule_ratio(branch, step, **kwargs), schedule_ratio(branch, step + 1, **kwargs)
        lr = base_lr_at(step, peak_lr=oc["peak_lr"], final_lr=oc["final_lr"],
                        warmup_steps=oc["warmup_steps"], decay_start_step=oc["decay_start_step"],
                        total_steps=oc["total_steps"])
        _set_lrs(optimizer, lr, q_now)
        sample_tensors = (
            monitor_enabled
            and (step % tensor_interval == 0 or step + 1 == oc["total_steps"]))
        tensor_pre_norms = _batched_frobenius_norms(trainable) if sample_tensors else None
        pre_update_norms = _batched_frobenius_norms(units)
        unit_pre_norms = pre_update_norms if sample_tensors and unit_monitor_enabled else None
        elrs = [(q_now * lr) / rms_from_norm(pre_update_norms[name], unit.numel())
                for name, unit in units.items()]
        targets = [lr / rms_from_norm(reference_norms[name], unit.numel())
                   for name, unit in units.items()]
        rel = [abs(actual - target) / max(abs(target), 1e-12)
               for actual, target in zip(elrs, targets)]
        before = ({n: p.detach().clone() for n, p in controlled.items()}
                  if step % lc["angular_interval"] == 0 else None)
        loss, acc = optimizer_step(model, optimizer, batches, accumulation, device)
        tensor_post_optimizer_norms = (
            _batched_frobenius_norms(trainable) if sample_tensors else None)
        unit_post_optimizer_norms = (
            _batched_frobenius_norms(units)
            if sample_tensors and unit_monitor_enabled else None)
        norm_mean, norm_max = project_controlled(
            controlled, reference_norms, q_next, cc["projection_eps"],
            split_fused_qkv=split_qkv)
        if sample_tensors:
            tensor_post_projection_norms = _batched_frobenius_norms(trainable)
            append_csv_rows(_tensor_monitor_rows(
                branch=branch, step=step, base_lr=lr,
                schedule_ratio_now=q_now, schedule_ratio_next=q_next,
                trainable=trainable, controlled=controlled,
                assignments=assignments, references=monitor_reference_norms,
                pre_norms=tensor_pre_norms,
                post_optimizer_norms=tensor_post_optimizer_norms,
                post_projection_norms=tensor_post_projection_norms),
                monitor_path)
            if unit_monitor_enabled:
                unit_post_projection_norms = _batched_frobenius_norms(units)
                append_csv_rows(_tensor_monitor_rows(
                    branch=branch, step=step, base_lr=lr,
                    schedule_ratio_now=q_now, schedule_ratio_next=q_next,
                    trainable=units, controlled=units,
                    assignments=unit_assignments,
                    references=control_unit_monitor_reference_norms,
                    pre_norms=unit_pre_norms,
                    post_optimizer_norms=unit_post_optimizer_norms,
                    post_projection_norms=unit_post_projection_norms),
                    unit_monitor_path)
        ema = loss if ema is None else lc["ema_beta"] * ema + (1 - lc["ema_beta"]) * loss
        angles = angular_steps(before, controlled) if before else {}
        for n, value in angles.items():
            cumulative[n] += value
        row = {
            "global_state_step": step + 1, "branch": branch, "train_loss_raw": loss,
            "train_loss_ema": ema, "train_top1": acc, "base_lr": lr,
            "controlled_lr": q_now * lr, "schedule_ratio": q_now,
            "controlled_frobenius_norm": sum(frobenius_norm(p) ** 2 for p in controlled.values()) ** .5,
            "target_norm_relative_error_mean": norm_mean,
            "target_norm_relative_error_max": norm_max,
            "tensorwise_elr_mean": statistics.mean(elrs), "tensorwise_elr_median": statistics.median(elrs),
            "tensorwise_elr_min": min(elrs), "tensorwise_elr_max": max(elrs),
            "elr_relative_error_mean": statistics.mean(rel), "elr_relative_error_max": max(rel),
            "gradient_finite": True,
            "angular_step_mean": statistics.mean(angles.values()) if angles else math.nan,
            "angular_step_median": statistics.median(angles.values()) if angles else math.nan,
        }
        append_csv(row, output / "metrics.csv")
        if probe_loader is not None and (step + 1) % lc["probe_interval"] == 0:
            metrics, _ = evaluate(model, probe_loader, device)
            append_csv({"global_state_step": step + 1, "branch": branch, "split": "probe",
                        **metrics}, output / "evaluation_metrics.csv")
        if val_loader is not None and (step + 1) % lc["val_interval"] == 0:
            metrics, _ = evaluate(model, val_loader, device)
            append_csv({"global_state_step": step + 1, "branch": branch, "split": "validation",
                        **metrics}, output / "evaluation_metrics.csv")
        if angles:
            for n, value in angles.items():
                append_csv({"global_state_step": step + 1, "branch": branch, "tensor": n,
                            "numel": controlled[n].numel(), "angular_step": value,
                            "cumulative_angular_step": cumulative[n]},
                           output / "angular_metrics.csv")
        if (step + 1) % lc["checkpoint_interval"] == 0 or step + 1 == oc["total_steps"]:
            save_checkpoint(output / f"checkpoint_step_{step+1:06d}.pt", model=model,
                            optimizer=optimizer, global_state_step=step + 1,
                            sampler_state=sampler.state_dict(), train_loss_ema=ema,
                            reference_norms=reference_norms, config=cfg,
                            controlled_names=control_units(
                                controlled, split_fused_qkv=split_qkv))
