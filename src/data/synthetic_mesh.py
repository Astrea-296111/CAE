from dataclasses import dataclass

import numpy as np


DESIGN_KEYS = (
    "thickness", "young_modulus", "poisson_ratio", "density",
    "reinforcement_factor", "section_scale", "geometry_scale_x",
    "geometry_scale_y", "geometry_scale_z", "local_morph_factor",
    "brace_probability", "load_force",
)
DESIGN_SCALES = np.array([1, 7e10, 0.3, 2700, 1, 1, 1, 1, 1, 0.1, 1, 1000])


@dataclass
class SyntheticMesh:
    pos: np.ndarray
    edges: np.ndarray
    boundary: np.ndarray
    left: np.ndarray
    right: np.ndarray
    force: np.ndarray
    thickness: np.ndarray
    region: np.ndarray
    design_parameters: dict
    mode: str

    @property
    def span(self):
        return float(self.pos[self.left, 1].mean() - self.pos[self.right, 1].mean())

    @property
    def torque(self):
        return float(np.dot(self.pos[:, 1], self.force))


def sample_design(rng, thickness_range=(0.8, 1.2)):
    low, high = map(float, thickness_range)
    if not 0 < low < high:
        raise ValueError("Thickness range must be positive and increasing")
    return dict(zip(DESIGN_KEYS, [
        rng.uniform(low, high), rng.uniform(5e10, 9e10), rng.uniform(0.25, 0.35),
        rng.uniform(2400, 3000), rng.uniform(1, 1.8), rng.uniform(0.8, 1.3),
        rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1),
        rng.uniform(-0.06, 0.06), rng.uniform(0.2, 0.9), rng.uniform(800, 1200),
    ]))


def generate_mesh(design, rng, mode="frame", grid_shape=(8, 4, 4)):
    if mode not in {"frame", "mesh"}:
        raise ValueError("mode must be frame or mesh")
    nx, ny, nz = map(int, grid_shape)
    if min(nx, ny, nz) < 2:
        raise ValueError("Every grid dimension must be at least two")
    if any(not np.isfinite(design[k]) for k in DESIGN_KEYS):
        raise ValueError("Design parameters must be finite")
    positive = [k for k in DESIGN_KEYS if k not in {"local_morph_factor", "brace_probability"}]
    if any(design[k] <= 0 for k in positive) or not 0 <= design["brace_probability"] <= 1:
        raise ValueError("Invalid material, geometry, load or brace parameters")
    if not 0 < design["poisson_ratio"] < 0.5:
        raise ValueError("Poisson ratio must lie in (0, 0.5)")
    indices = np.array(list(np.ndindex(nx, ny, nz)))
    xyz = indices / np.array([nx - 1, ny - 1, nz - 1])
    pos = xyz * np.array([4, 1.6, 1.2])
    pos[:, 1] -= 0.8
    pos *= np.array([design[f"geometry_scale_{c}"] for c in "xyz"])
    taper = np.sin(np.pi * xyz[:, 0])
    pos[:, 1] += design["local_morph_factor"] * taper * np.cos(np.pi * xyz[:, 2])
    pos[:, 2] += design["local_morph_factor"] * taper * xyz[:, 1]
    pos += rng.normal(0, 0.003, pos.shape) * taper[:, None]

    def node(i, j, k):
        return (i * ny + j) * nz + k

    edges = set()
    for i, j, k in indices:
        a = node(i, j, k)
        if i + 1 < nx:
            edges.add((a, node(i + 1, j, k)))
        cross_section = mode == "mesh" or i in {0, nx - 1} or i % 2 == 0
        if cross_section and j + 1 < ny:
            edges.add((a, node(i, j + 1, k)))
        if cross_section and k + 1 < nz:
            edges.add((a, node(i, j, k + 1)))
        if i + 1 < nx and j + 1 < ny and rng.random() < design["brace_probability"]:
            edges.add((a, node(i + 1, j + 1, k)))
        if i + 1 < nx and k + 1 < nz and rng.random() < design["brace_probability"]:
            edges.add((a, node(i + 1, j, k + 1)))
    boundary = indices[:, 0] == nx - 1
    left = (indices[:, 0] == 0) & (indices[:, 1] == ny - 1) & (indices[:, 2] == 0)
    right = (indices[:, 0] == 0) & (indices[:, 1] == 0) & (indices[:, 2] == 0)
    force = design["load_force"] * (left.astype(float) - right.astype(float))
    region = np.minimum((xyz[:, 0] * 3).astype(int), 2)
    thickness = design["thickness"] * (
        1 + 0.15 * np.exp(-((xyz[:, 0] - 0.5) / 0.2) ** 2)
        * (design["reinforcement_factor"] - 1)
    )
    return SyntheticMesh(pos, np.array(sorted(edges), dtype=np.int64), boundary,
                         left, right, force, thickness, region, dict(design), mode)


def edge_stiffness(mesh):
    a, b = mesh.edges.T
    delta = mesh.pos[b] - mesh.pos[a]
    length = np.linalg.norm(delta, axis=1)
    if np.any(length <= 1e-9):
        raise ValueError("Zero-length structural edge")
    direction = delta / length[:, None]
    d = mesh.design_parameters
    midpoint_x = (mesh.pos[a, 0] + mesh.pos[b, 0]) / (2 * mesh.pos[:, 0].max())
    reinforcement = 1 + (d["reinforcement_factor"] - 1) * np.exp(
        -((midpoint_x - 0.5) / 0.2) ** 2
    )
    thickness = (mesh.thickness[a] + mesh.thickness[b]) / 2
    orientation = 0.2 + direction[:, 2] ** 2 + 0.4 * direction[:, 1] ** 2
    return (2e6 * (d["young_modulus"] / 7e10) * thickness ** 2.2
            * d["section_scale"] ** 1.3 * reinforcement * orientation
            / ((1 + d["poisson_ratio"]) * length))
