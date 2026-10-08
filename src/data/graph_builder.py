import numpy as np
import torch
from torch_geometric.data import Data

from .synthetic_mesh import DESIGN_KEYS, DESIGN_SCALES, edge_stiffness


NODE_FEATURES = (
    "x_m", "y_m", "z_m", "normalized_x", "normalized_y", "normalized_z",
    "thickness_mm", "young_modulus_over_70GPa", "poisson_ratio", "density_over_2700",
    "is_boundary", "is_load_node", "load_x_over_1000N", "load_y_over_1000N",
    "load_z_over_1000N", "region_0", "region_1", "region_2",
)
EDGE_FEATURES = ("dx_m", "dy_m", "dz_m", "distance_m", "direction_x",
                 "direction_y", "direction_z", "stiffness_proxy_over_1e7Nm")


def build_graph(mesh, result=None, sample_id="synthetic"):
    pos = mesh.pos
    normalized = (pos - pos.min(0)) / np.maximum(np.ptp(pos, axis=0), 1e-9)
    n = len(pos)
    d = mesh.design_parameters
    features = np.column_stack([
        pos, normalized, mesh.thickness,
        np.full(n, d["young_modulus"] / 7e10), np.full(n, d["poisson_ratio"]),
        np.full(n, d["density"] / 2700), mesh.boundary,
        mesh.left | mesh.right, np.zeros(n), np.zeros(n), mesh.force / 1000,
        np.eye(3)[mesh.region],
    ])
    undirected = mesh.edges.T
    edges = np.concatenate([undirected, undirected[::-1]], axis=1)
    delta = pos[edges[1]] - pos[edges[0]]
    length = np.linalg.norm(delta, axis=1)
    weights = np.tile(edge_stiffness(mesh), 2)
    edge_features = np.column_stack([delta, length, delta / length[:, None], weights / 1e7])
    graph = Data(
        x=torch.tensor(features, dtype=torch.float32),
        pos=torch.tensor(pos, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        boundary=torch.tensor(mesh.boundary),
        left_mask=torch.tensor(mesh.left), right_mask=torch.tensor(mesh.right),
        force=torch.tensor(mesh.force, dtype=torch.float32),
        span=torch.tensor([mesh.span], dtype=torch.float32),
        torque=torch.tensor([mesh.torque], dtype=torch.float32),
        design_parameters=torch.tensor(
            [[d[k] for k in DESIGN_KEYS]] / DESIGN_SCALES, dtype=torch.float32
        ),
        mesh_mode=torch.tensor([int(mesh.mode == "mesh")]),
        sample_id=sample_id, provenance="synthetic-only",
    )
    if result is not None:
        graph.y_k = torch.tensor([result.stiffness], dtype=torch.float32)
        graph.y_u = torch.tensor(result.displacement, dtype=torch.float32)
    graph.validate(raise_on_error=True)
    return graph
