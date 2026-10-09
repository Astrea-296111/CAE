"""Label-free candidate manifests with reproducible topology seeds."""

import json
from pathlib import Path

import numpy as np

from .synthetic_mesh import DESIGN_KEYS, generate_mesh, sample_design


def generate_candidates(config, count=100, seed=2026, ood_fraction=0.2):
    if count < 1 or not 0 <= ood_fraction <= 1:
        raise ValueError("count must be positive and ood_fraction must be in [0, 1]")
    rng = np.random.default_rng(seed)
    settings = config["dataset"]
    ood_count = round(count * ood_fraction)
    ood_indices = set(rng.choice(count, size=ood_count, replace=False).tolist())
    candidates = []
    for i in range(count):
        interval = settings["ood_thickness_range"] if i in ood_indices else settings["thickness_range"]
        candidates.append({
            "sample_id": f"design-{i + 1:04d}",
            "mesh_seed": int(rng.integers(0, 2**31)),
            "mode": str(rng.choice(settings.get("modes", ["frame", "mesh"]))),
            "grid_shape": list(settings.get("grid_shape", [8, 4, 4])),
            "parameters": sample_design(rng, interval),
        })
    return {"schema_version": 1, "provenance": "synthetic-only",
            "generation": {"seed": seed, "thickness_ood_count": ood_count},
            "candidates": candidates}


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Expected candidate manifest schema_version=1")
    if manifest.get("provenance") != "synthetic-only":
        raise ValueError("This candidate adapter accepts synthetic-only inputs")
    if set(manifest) - {"schema_version", "provenance", "generation", "candidates"}:
        raise ValueError("Unknown manifest fields")
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest requires a nonempty candidates list")
    ids = [row.get("sample_id") if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(s, str) or not s.strip() for s in ids):
        raise ValueError("Every candidate requires a nonempty string sample_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id in candidate manifest")
    return manifest


def load_candidates(path):
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def candidate_mesh(row, max_nodes=10000):
    expected = {"sample_id", "mesh_seed", "mode", "grid_shape", "parameters"}
    if set(row) != expected:
        raise ValueError("Candidate fields must be sample_id, mesh_seed, mode, grid_shape, parameters")
    seed = row["mesh_seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("mesh_seed must be an integer in [0, 2**32)")
    shape = row["grid_shape"]
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(v) is not int or v < 2 or v > max_nodes for v in shape)
            or int(np.prod(shape, dtype=object)) > max_nodes):
        raise ValueError(f"grid_shape must have three integer dimensions and at most {max_nodes} nodes")
    parameters = row["parameters"]
    if not isinstance(parameters, dict) or set(parameters) != set(DESIGN_KEYS):
        raise ValueError("Candidate parameters do not match the design schema")
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in parameters.values()):
        raise ValueError("Candidate parameters must be finite numbers")
    return generate_mesh(parameters, np.random.default_rng(seed), row["mode"], shape)
