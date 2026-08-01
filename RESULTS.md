# V6 results

## Formal decision

The registered threshold is `EMA-MAE <= 0.03` for every non-constant branch.
Engineering validity passed, but the scientific collapse gate failed.

| branch | raw-loss MAE | EMA-loss MAE | max EMA gap | decision |
|---|---:|---:|---:|---|
| `linear_up` | 0.0714760 | 0.0371595 | 0.1265708 | FAIL |
| `linear_down` | 0.2363062 | 0.2282249 | 0.6379953 | FAIL |
| `cyclic` | 0.2289799 | 0.2258968 | 0.7377430 | FAIL |

- `validity_passed = true`
- `collapse_passed = false`
- `passed = false`
- failing branches: `linear_up`, `linear_down`, `cyclic`

Compared with V5, the EMA-MAE ratios `V6/V5` are approximately `1.107`,
`1.005`, and `1.013`. Exact final-gain norm matching therefore produced no
collapse improvement.

## Validation at step 20,000

| branch | validation loss | top-1 |
|---|---:|---:|
| `constant` | 3.9873151 | 35.09% |
| `linear_up` | 4.0727176 | 34.44% |
| `linear_down` | 3.4129210 | 36.41% |
| `cyclic` | 3.6291788 | 36.56% |

The lower validation loss of some non-constant branches does not constitute
loss collapse; the gate compares the paired training-loss EMA trajectories.

## Engineering validity

All fail-closed audits passed:

| audit | coverage | result |
|---|---:|---|
| effective-LR / target-norm validity | max errors `2.94e-7` / `2.51e-7` | PASS |
| named tensors | 445,008 rows; 127 tensors = 100 controlled + 27 uncontrolled | PASS |
| split control units | 518,592 rows; 148 controlled units | PASS |
| final `norm.weight` alignment | 70,000 rows; 17,500 steps per branch | PASS |
| checkpoint provenance | four final checkpoints, config, code archive, prefix | PASS |

For the V6-specific alignment audit:

- tensor: `norm.weight` (192 scalars)
- reference: `constant`
- maximum post-alignment relative error: `2.166999e-7`
- maximum cross-branch pre-update norm relative error: `2.166999e-7`
- maximum cross-branch actual-LR relative error: `0.0`
- `constant` was never projected; exactly 52,500 non-constant updates were
  projected.

The ordinary named-tensor monitor samples after the wrapped optimizer step, so
for this one tensor its field named `post_optimizer_pre_projection` is already
post-alignment. Use the dedicated `reference_alignment_metrics.csv` and
`reference_alignment_audit.json` for the intervention-before/after evidence.

## Scope and interpretation

V6 retains the V5 q(t)-control policy: patch projection weight/bias, CLS token,
position embedding, and every block's split Q/K/V weight/bias, attention output
projection weight/bias, FC1 weight/bias, and FC2 weight/bias. This is 100 named
tensors and 148 independent units. The 25 RMSNorm gains and classifier-head
weight/bias remain outside q(t) control.

Only the final RMSNorm gain receives the additional radial intervention. It
still uses the base LR, takes its own AdamW update, retains its own direction,
and keeps its own AdamW moments.

Because the intervention was numerically exact yet the V5-sized loss gaps
remained, the V5 correlation between final-gain norm drift and loss similarity
was not a sufficient causal explanation. The result only rejects the narrow
claim that this scalar trajectory alone is enough. Candidate mechanisms still
include the gain direction, the other 24 RMSNorm gains, optimizer moments, and
the non-scale-invariant Q/K/V and FC1 pathways controlled in V5/V6.

## Provenance

- source commit: `30984678281830ab61b621e7ec71c85f5f1d9702`
- uploaded code archive SHA256:
  `f3f8057eb7e8deb7a057842efaaf2932bad1b0087808d9dba9fadf30b2d189a6`
- registered V5 prefix SHA256:
  `48789c6d64c46e1bdbbd8922db01ec4472d4ca72ce2911c6f27a5a6bfed66773`
- resolved-config SHA256:
  `746e4f4e077c50ac6650b0562c8c09c269425c18ba3f722adfb0c1c339e23b3d`
- full archive: `callapse_v6_results_20260802.tar.gz`
- archive size: `1,941,078,723` bytes
- archive SHA256:
  `fd358ac99b9d5239a7258c4e8f72c532f4e3df400ede3d0e40cd47d6e77722ab`
- extracted payload: 415 files, `2,460,247,254` bytes

The full archive is stored next to the local repository; the GitHub repository
contains code plus the compact figures, tables, and audit JSON/CSV files.
