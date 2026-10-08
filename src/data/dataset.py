from pathlib import Path

import numpy as np
import torch

from .graph_builder import build_graph
from .split import split_indices, validate_thickness_ood
from .synthetic_mesh import generate_mesh, sample_design
from .synthetic_solver import SyntheticPhysicsSolver


class SyntheticFEMDatasetGenerator:
    def __init__(self, config):
        self.config = config
        self.dataset_config = config["dataset"]
        self.solver = SyntheticPhysicsSolver(self.dataset_config.get("noise_std", 0))

    def sample_mesh(self, rng, thickness_range=None):
        c = self.dataset_config
        design = sample_design(rng, thickness_range or c["thickness_range"])
        modes = c.get("modes", ["frame", "mesh"])
        if not modes:
            raise ValueError("At least one mesh mode is required")
        return generate_mesh(design, rng, str(rng.choice(modes)), c.get("grid_shape", [8, 4, 4]))

    def generate(self):
        c = self.dataset_config
        validate_thickness_ood(c["thickness_range"], c["ood_thickness_range"])
        sizes = [int(c[f"num_{s}"]) for s in ("train", "val", "test", "ood")]
        if min(sizes) < 1:
            raise ValueError("All four split sizes must be positive")
        seed = int(self.config.get("seed", 42))
        rng = np.random.default_rng(seed)
        graphs = []
        for i in range(sum(sizes[:3])):
            mesh = self.sample_mesh(rng)
            graphs.append(build_graph(mesh, self.solver.run(mesh, rng), f"id-{seed}-{i}"))
        split = split_indices(len(graphs), sizes[0], sizes[1], seed)
        dataset = {name: [graphs[int(i)] for i in indices] for name, indices in split.items()}
        ood_rng = np.random.default_rng(seed + 100000)
        dataset["ood"] = []
        for i in range(sizes[3]):
            mesh = self.sample_mesh(ood_rng, c["ood_thickness_range"])
            dataset["ood"].append(build_graph(
                mesh, self.solver.run(mesh, ood_rng), f"ood-{seed}-{i}"
            ))
        dataset["metadata"] = {
            "provenance": "synthetic-only", "seed": seed, "config": self.config,
            "units": {"position": "m", "thickness": "mm", "force": "N",
                      "displacement": "m", "angle": "rad", "stiffness": "N*m/rad"},
            "solver": "scalar transverse elastic network; not full FEM",
        }
        return dataset


def save_dataset(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, path)


def load_dataset(path):
    # PyG Data objects require pickle; only load locally generated trusted files.
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    if dataset.get("metadata", {}).get("provenance") != "synthetic-only":
        raise ValueError("Expected a synthetic-only dataset")
    return dataset
