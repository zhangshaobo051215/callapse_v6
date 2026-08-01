from __future__ import annotations

from . import train_branch as base_train_branch
from .reference_alignment import build_constant_reference_norm_aligner
from .schedules import base_lr_at, schedule_ratio


_ORIGINAL_TRAIN_BRANCH = base_train_branch.train_branch
_ORIGINAL_OPTIMIZER_STEP = base_train_branch.optimizer_step


def train_branch(
    cfg,
    branch,
    model,
    optimizer,
    controlled,
    sampler,
    loader_factory,
    device,
    output,
    start_step,
    ema,
    reference_norms,
    probe_loader=None,
    val_loader=None,
    monitor_reference_norms=None,
    control_unit_monitor_reference_norms=None,
):
    """Run the stock V5 branch loop with one post-Adam final-gain hook.

    The hook executes immediately after ``optimizer.step`` and before the
    ordinary q(t) projection.  ``norm.weight`` is not in the q-controlled set,
    so the later V5 projection cannot alter it.  The dedicated alignment CSV
    retains the true pre-alignment post-optimizer norm at every step.
    """
    aligner = build_constant_reference_norm_aligner(
        cfg,
        branch=branch,
        model=model,
        optimizer=optimizer,
        controlled=controlled,
        output=output,
        start_step=start_step,
    )
    if aligner is None:
        return _ORIGINAL_TRAIN_BRANCH(
            cfg, branch, model, optimizer, controlled, sampler,
            loader_factory, device, output, start_step, ema, reference_norms,
            probe_loader, val_loader,
            monitor_reference_norms=monitor_reference_norms,
            control_unit_monitor_reference_norms=(
                control_unit_monitor_reference_norms),
        )

    state = {"step": int(start_step)}
    oc, cc = cfg["optimizer"], cfg["control"]

    def aligned_optimizer_step(model_arg, optimizer_arg, batches,
                               accumulation, device_arg):
        step = state["step"]
        pre_update_norm = aligner.pre_update_norm()
        loss, accuracy = _ORIGINAL_OPTIMIZER_STEP(
            model_arg, optimizer_arg, batches, accumulation, device_arg)
        base_lr = base_lr_at(
            step,
            peak_lr=oc["peak_lr"],
            final_lr=oc["final_lr"],
            warmup_steps=oc["warmup_steps"],
            decay_start_step=oc["decay_start_step"],
            total_steps=oc["total_steps"],
        )
        q_now = schedule_ratio(
            branch,
            step,
            control_start_step=oc["control_start_step"],
            total_steps=oc["total_steps"],
            cyclic_period_steps=cc["cyclic_period_steps"],
            cyclic_amplitude=cc["cyclic_amplitude"],
            linear_up_final=cc["linear_up_final"],
            linear_down_final=cc["linear_down_final"],
        )
        aligner.apply(
            update_start_step=step,
            base_lr=base_lr,
            schedule_ratio=q_now,
            pre_update_norm=pre_update_norm,
        )
        state["step"] = step + 1
        return loss, accuracy

    previous_optimizer_step = base_train_branch.optimizer_step
    if previous_optimizer_step is not _ORIGINAL_OPTIMIZER_STEP:
        raise RuntimeError("optimizer_step was already patched")
    base_train_branch.optimizer_step = aligned_optimizer_step
    try:
        return _ORIGINAL_TRAIN_BRANCH(
            cfg, branch, model, optimizer, controlled, sampler,
            loader_factory, device, output, start_step, ema, reference_norms,
            probe_loader, val_loader,
            monitor_reference_norms=monitor_reference_norms,
            control_unit_monitor_reference_norms=(
                control_unit_monitor_reference_norms),
        )
    finally:
        base_train_branch.optimizer_step = previous_optimizer_step


def install_into(run_pipeline_module) -> None:
    run_pipeline_module.train_branch = train_branch
