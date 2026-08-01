# V6 results

Status: implementation and local tests complete; registered AutoDL run pending.

V6 keeps the exact V5 architecture and q(t)-controlled scope while aligning
only the final RMSNorm `norm.weight` Frobenius-norm trajectory to `constant`.
The final report will include:

- EMA-MAE versus `constant` for `linear_up`, `linear_down`, and `cyclic`;
- the fixed `0.03` collapse gate;
- named-tensor, control-unit, and constant-reference alignment audits;
- the complete norm/LR monitoring figures and final checkpoint comparison.
