from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .synthetic_mesh import edge_stiffness


@dataclass
class SolverResult:
    stiffness: float
    displacement: np.ndarray
    torque: float
    angle: float
    relative_residual: float
    provenance: str = "synthetic-only"


class SyntheticPhysicsSolver:
    """Scalar transverse spring network, not a shell/beam FEM implementation."""

    def __init__(self, noise_std=0.0):
        if noise_std < 0:
            raise ValueError("noise_std must be nonnegative")
        self.noise_std = noise_std

    def run(self, mesh, rng=None):
        weights = edge_stiffness(mesh)
        if self.noise_std:
            if rng is None:
                raise ValueError("A seeded generator is required for label noise")
            weights *= np.exp(rng.normal(0, self.noise_std))
        a, b = mesh.edges.T
        rows = np.concatenate([a, b, a, b])
        cols = np.concatenate([a, b, b, a])
        values = np.concatenate([weights, weights, -weights, -weights])
        stiffness_matrix = coo_matrix(
            (values, (rows, cols)), shape=(len(mesh.pos), len(mesh.pos))
        ).tocsr()
        free = ~mesh.boundary
        reduced = stiffness_matrix[free][:, free]
        displacement = np.zeros(len(mesh.pos))
        displacement[free] = spsolve(reduced, mesh.force[free])
        residual = np.linalg.norm(reduced @ displacement[free] - mesh.force[free])
        residual /= max(np.linalg.norm(mesh.force[free]), 1e-12)
        angle = float((displacement[mesh.left].mean()
                       - displacement[mesh.right].mean()) / mesh.span)
        if not np.all(np.isfinite(displacement)) or angle <= 0 or residual > 1e-7:
            raise RuntimeError("Invalid or disconnected synthetic elastic network")
        return SolverResult(mesh.torque / angle, displacement[:, None],
                            mesh.torque, angle, float(residual))
