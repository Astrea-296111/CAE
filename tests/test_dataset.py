import numpy as np
import pytest
import torch

from src.data.dataset import SyntheticFEMDatasetGenerator, load_dataset, save_dataset
from src.data.synthetic_mesh import generate_mesh, sample_design
from src.data.synthetic_solver import SyntheticPhysicsSolver


def test_seed_split_and_ood(config, tmp_path):
    first = SyntheticFEMDatasetGenerator(config).generate()
    second = SyntheticFEMDatasetGenerator(config).generate()
    ids = [g.sample_id for split in ("train", "val", "test", "ood") for g in first[split]]
    assert len(ids) == len(set(ids))
    for split in ("train", "val", "test", "ood"):
        for a, b in zip(first[split], second[split]):
            assert torch.equal(a.x, b.x)
            assert torch.equal(a.y_k, b.y_k)
            thickness = a.design_parameters[0, 0].item()
            assert (1.25 <= thickness <= 1.4) if split == "ood" else (0.8 <= thickness <= 1.2)
    path = tmp_path / "data.pt"
    save_dataset(first, path)
    assert len(load_dataset(path)["train"]) == 4


@pytest.mark.parametrize("mode", ["frame", "mesh"])
def test_solver_equilibrium_boundary_and_load_scaling(mode):
    rng = np.random.default_rng(4)
    design = sample_design(rng)
    mesh = generate_mesh(design, np.random.default_rng(17), mode)
    result = SyntheticPhysicsSolver().run(mesh)
    assert result.relative_residual < 1e-9
    assert np.all(result.displacement[mesh.boundary] == 0)
    assert result.stiffness == pytest.approx(mesh.torque / result.angle)
    assert mesh.force.sum() == pytest.approx(0)
    design["load_force"] *= 2
    doubled = generate_mesh(design, np.random.default_rng(17), mode)
    other = SyntheticPhysicsSolver().run(doubled)
    assert other.stiffness == pytest.approx(result.stiffness)
    assert other.displacement == pytest.approx(2 * result.displacement)
    design["thickness"] *= 1.2
    thick = generate_mesh(design, np.random.default_rng(17), mode)
    assert SyntheticPhysicsSolver().run(thick).stiffness > other.stiffness
