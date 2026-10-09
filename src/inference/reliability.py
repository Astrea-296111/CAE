from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityPolicy:
    max_physics_residual: float = 0.25
    domain_margin: float = 0.02
    require_displacement: bool = True
    max_nodes: int = 10000

    def __post_init__(self):
        if not np.isfinite(self.max_physics_residual) or self.max_physics_residual < 0:
            raise ValueError("max_physics_residual must be finite and nonnegative")
        if not np.isfinite(self.domain_margin) or not 0 <= self.domain_margin <= 0.1:
            raise ValueError("domain_margin must be in [0, 0.1]")
        if type(self.max_nodes) is not int or self.max_nodes < 8:
            raise ValueError("max_nodes must be an integer >= 8")
        if type(self.require_displacement) is not bool:
            raise ValueError("require_displacement must be boolean")

    def to_dict(self):
        return asdict(self)


def assess_domain(mesh, domain, policy):
    reasons = []
    for key, (low, high) in domain["parameters"].items():
        # Training features were serialized as float32; tolerate their rounding,
        # including fixed parameters whose empirical span is zero.
        padding = max(policy.domain_margin * (high - low), 1e-6 * max(abs(low), abs(high), 1e-12))
        value = mesh.design_parameters[key]
        if not low - padding <= value <= high + padding:
            reasons.append(f"ood:{key}")
    if mesh.mode not in domain["mesh_modes"]:
        reasons.append("ood:mesh_mode")
    if len(mesh.pos) not in domain["node_counts"]:
        reasons.append("ood:node_count")
    return reasons


def assess_prediction(mesh, stiffness, displacement, domain, policy):
    reasons = assess_domain(mesh, domain, policy)
    if not np.isfinite(stiffness) or stiffness <= 0:
        reasons.append("invalid_stiffness")
    angle, residual = None, None
    if displacement is None:
        if policy.require_displacement:
            reasons.append("missing_displacement")
    else:
        u = np.asarray(displacement)
        if u.shape != (len(mesh.pos), 1) or not np.isfinite(u).all():
            reasons.append("invalid_displacement")
        else:
            if not np.all(u[mesh.boundary] == 0):
                reasons.append("boundary_displacement")
            angle = float((u[mesh.left].mean() - u[mesh.right].mean()) / mesh.span)
            if angle <= 0:
                reasons.append("nonpositive_angle")
            if np.isfinite(stiffness):
                value = float(abs(stiffness * angle / mesh.torque - 1))
                if np.isfinite(value):
                    residual = value
                if residual is None or residual > policy.max_physics_residual:
                    reasons.append("physics_residual")
    return {"reasons": reasons, "angle_rad": angle, "physics_residual": residual}
