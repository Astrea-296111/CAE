"""Versioned preprocessing contract shared by training and inference."""

import hashlib
import json

import numpy as np

from .graph_builder import EDGE_FEATURES, NODE_FEATURES
from .synthetic_mesh import DESIGN_KEYS, DESIGN_SCALES


def feature_schema():
    return {
        "version": 1,
        "node_features": list(NODE_FEATURES),
        "edge_features": list(EDGE_FEATURES),
        "design_features": list(DESIGN_KEYS),
        "design_scales": DESIGN_SCALES.tolist(),
        "units": {"position": "m", "thickness": "mm", "force": "N",
                  "displacement": "m", "angle": "rad", "stiffness": "N*m/rad"},
        "displacement_components": ["uz"],
        "graph_builder": "synthetic-elastic-network-v1",
    }


def schema_hash():
    return hashlib.sha256(json.dumps(feature_schema(), sort_keys=True).encode()).hexdigest()


def fit_training_domain(graphs):
    """Empirical envelope from training inputs only; not a confidence interval."""
    values = np.concatenate([g.design_parameters.numpy() for g in graphs]) * DESIGN_SCALES
    return {
        "source": "training-inputs-only", "sample_count": len(graphs),
        "parameters": {key: [float(values[:, i].min()), float(values[:, i].max())]
                       for i, key in enumerate(DESIGN_KEYS)},
        "mesh_modes": sorted({"mesh" if g.mesh_mode.item() else "frame" for g in graphs}),
        "node_counts": sorted({g.num_nodes for g in graphs}),
    }


def validate_checkpoint_contract(checkpoint, require_contract=False):
    version = checkpoint.get("checkpoint_version")
    if version is None:
        if require_contract:
            raise ValueError("Checkpoint has no feature contract; retrain before batch screening")
        return  # v0 checkpoints remain readable for historical evaluation.
    if version != 1 or checkpoint.get("feature_schema") != feature_schema():
        raise ValueError("Checkpoint feature schema does not match the graph builder")
    if checkpoint.get("feature_schema_sha256") != schema_hash():
        raise ValueError("Checkpoint feature schema hash mismatch")
    domain = checkpoint.get("training_domain", {})
    if domain.get("source") != "training-inputs-only":
        raise ValueError("Checkpoint is missing its training domain")
    if set(domain.get("parameters", {})) != set(DESIGN_KEYS):
        raise ValueError("Checkpoint training domain parameters are incomplete")
    for bounds in domain["parameters"].values():
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("Invalid training domain bounds")
    if not domain.get("mesh_modes") or not domain.get("node_counts"):
        raise ValueError("Checkpoint training domain geometry is incomplete")
