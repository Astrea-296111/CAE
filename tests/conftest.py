import numpy as np
import pytest
import torch

from src.data.graph_builder import build_graph
from src.data.synthetic_mesh import generate_mesh, sample_design
from src.data.synthetic_solver import SyntheticPhysicsSolver
from src.utils.config import load_config, merge_config


@pytest.fixture
def config():
    return merge_config(load_config("configs/demo.yaml"), {
        "dataset": {"num_train": 4, "num_val": 2, "num_test": 2, "num_ood": 2},
        "model": {"hidden_dim": 16, "message_passing_steps": 2},
    })


@pytest.fixture
def graph():
    torch.set_num_threads(1)
    rng = np.random.default_rng(12)
    mesh = generate_mesh(sample_design(rng), rng)
    return build_graph(mesh, SyntheticPhysicsSolver().run(mesh))
