# Executed Synthetic Baseline Comparison

CPU demo, seed 42, 120 train / 30 validation / 30 test / 30 OOD, at most 30 epochs.
All values below are from `results/comparison.csv`; they are not industrial results.

| Model | ID R² | ID MAPE % | OOD R² | OOD MAPE % |
|---|---:|---:|---:|---:|
| Linear (log target) | 0.8913 | 8.70 | 0.7400 | 13.74 |
| Random Forest | 0.7502 | 13.62 | -0.8550 | 32.94 |
| MLP | 0.8668 | 9.14 | 0.7522 | 12.74 |
| GCN | 0.4996 | 15.66 | -1.2913 | 37.55 |
| MeshGraphNet-style | 0.9527 | 6.19 | 0.5752 | 11.72 |
| Multi-task | 0.9589 | 5.22 | 0.6273 | 11.54 |
| Multi-task + Physics | 0.9645 | 4.93 | 0.6478 | 11.16 |

Parameter models cannot observe the realized bracing topology or mesh mode. This is an
information difference as well as an architecture comparison. Linear regression performs
well because the multiplicative synthetic law is easier to learn in log space. One seed and
30 ID test samples do not establish a stable ranking of algorithms.
