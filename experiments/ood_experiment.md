# Executed Synthetic Thickness OOD

Train/validation/ID thickness: [0.8, 1.2] mm. OOD thickness: [1.25, 1.4] mm.
Other variables use the same sampling ranges. This is parameter extrapolation, not a
new vehicle family or an entirely different structural topology.

The physics multi-task model changes from ID R² 0.9645 to OOD R² 0.6478; MAPE
changes from 4.93% to 11.16%. The MLP has OOD R² 0.7522 with 12.74% MAPE.
These metrics emphasize different aspects of error and do not justify a universal winner.
The forest and GCN have negative OOD R², exposing substantial extrapolation failure.

Follow-up protocol: freeze these splits, add multiple training/data seeds, test geometry,
bracing and mesh-resolution shifts independently, and report uncertainty calibration.
Do not tune on this OOD set and continue calling it an independent test.
