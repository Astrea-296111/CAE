# Executed Synthetic Ablation and Further Template

Same data file and 30-epoch budget; details are in `results/metrics.json`.

| Configuration | ID R² | ID displacement RMSE (m) | Mean torque consistency residual |
|---|---:|---:|---:|
| MeshGraphNet stiffness only | 0.9527 | N/A | N/A |
| Multi-task | 0.9589 | 8.925e-6 | 0.1009 |
| Multi-task + Physics | 0.9645 | 9.221e-6 | 0.1037 |

The physics version improves stiffness error slightly in this run, but neither displacement
RMSE nor ID torque consistency improves. Do not claim all physics metrics improved.
Validation-based model selection, a small weight and a short budget can affect the outcome;
multiple controlled runs are required to understand it.

Further experiment template, not executed:

| Change | Seeds | ID metrics | OOD metrics | Top-K | Runtime |
|---|---|---|---|---|---|
| No edge attributes | 42-46 | Pending | Pending | Pending | Pending |
| 1/3/6/10 processing steps | 42-46 | Pending | Pending | Pending | Pending |
| Physics weight 0/0.001/0.01/0.1 | 42-46 | Pending | Pending | Pending | Pending |
| Parameter model + topology statistics | 42-46 | Pending | Pending | Pending | Pending |
