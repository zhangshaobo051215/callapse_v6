# callapse_v6

V6 is a one-variable causal ablation built directly on `callapse_v5`.
It tests whether the schedule-dependent drift of the final RMSNorm gain
(`norm.weight`) is responsible for V5's failed loss collapse.

## Intervention

The model, Tiny-ImageNet-200 data order, optimizer state, shared step-2500
prefix, four schedules, and the V5 q(t)-controlled scope are unchanged.

- q(t)-controlled named tensors: 100 (148 independent control units after
  splitting fused Q/K/V weight and bias).
- Ordinary/uncontrolled named tensors: 27.
- The final RMSNorm gain remains outside q(t) control and keeps the base LR.
- `constant` trains normally and records its post-AdamW `norm.weight`
  Frobenius norm at every step.
- After each AdamW update, `linear_up`, `linear_down`, and `cyclic` are radially
  projected so that only
  `||norm.weight||_F` equals the same-step `constant` value.
- The 192-dimensional gain direction and AdamW moments are not copied or
  modified.

Thus V6 aligns the final gain's scalar norm and LR without changing any other
V5 intervention. It does **not** q-control the classifier head.

## Reproduce

The registered V5 RMSNorm prefix must be available at the sibling path used by
`go_v6.sh`. Then run:

```bash
bash go_v6.sh
```

The launcher runs the full test suite, verifies the V5 prefix SHA256, executes
the four branches in reference-first order, generates all analyses, and checks
three fail-closed audits before applying the fixed collapse threshold.

## Outputs

The full run is written to:

```text
outputs/expanded_block_affine_rmsnorm_finalnorm_v6_tensor_monitoring/
```

Important artifacts include:

- `analysis/figure1_vit.png`: four-branch collapse overview;
- `analysis/collapse_metrics.csv`: numerical loss-collapse metrics;
- `analysis/reference_alignment/final_norm_weight_alignment.png`: exact V6
  intervention diagnostic;
- `analysis/reference_alignment_audit.json`: all-step alignment/LR audit;
- `analysis/tensor_monitoring/per_tensor/`: every named tensor;
- `analysis/control_unit_monitoring/`: all 148 independent q-controlled units.

See `RESULTS.md` for the completed outcome after the registered run.
