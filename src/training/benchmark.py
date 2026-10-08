import platform
from time import perf_counter

import numpy as np
import torch

from .evaluator import predict_graphs
from ..data.dataset import SyntheticFEMDatasetGenerator
from ..data.graph_builder import build_graph
from ..utils.config import resolve_path
from ..utils.logging import write_json


def benchmark(model, config, count=100, repeats=5, output_dir="outputs/benchmark"):
    if count < 1 or repeats < 1:
        raise ValueError("count and repeats must be positive")
    generator = SyntheticFEMDatasetGenerator(config)
    rng = np.random.default_rng(config.get("seed", 42) + 81234)
    meshes = [generator.sample_mesh(rng) for _ in range(count)]
    started = perf_counter()
    graphs = [build_graph(mesh) for mesh in meshes]
    build_seconds = perf_counter() - started
    batch_size = config["training"]["batch_size"]
    use_cuda = next(model.parameters()).is_cuda

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    predict_graphs(model, graphs, batch_size)
    generator.solver.run(meshes[0], rng)
    solver_times, inference_times = [], []
    for _ in range(repeats):
        started = perf_counter()
        for mesh in meshes:
            generator.solver.run(mesh, rng)
        solver_times.append(perf_counter() - started)
        sync()
        started = perf_counter()
        predict_graphs(model, graphs, batch_size)
        sync()
        inference_times.append(perf_counter() - started)
    solver_median, inference_median = float(np.median(solver_times)), float(np.median(inference_times))
    result = {
        "provenance": "synthetic-only", "count": count, "repeats": repeats,
        "synthetic_solver_seconds": solver_times, "gnn_inference_seconds": inference_times,
        "solver_median_seconds": solver_median, "inference_median_seconds": inference_median,
        "solver_over_inference_ratio": solver_median / inference_median,
        "graph_build_seconds": build_seconds,
        "ai_with_graph_build_seconds": inference_median + build_seconds,
        "device": str(next(model.parameters()).device), "torch_version": torch.__version__,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(), "batch_size": batch_size,
        "scope": "Shared mesh generation excluded; inference includes graph batching and CPU output transfer.",
        "disclaimer": "Synthetic solver vs AI only. No OptiStruct or industrial speedup claim.",
    }
    write_json(resolve_path(output_dir) / "benchmark.json", result)
    return result
