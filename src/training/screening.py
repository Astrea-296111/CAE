from time import perf_counter

import numpy as np
import pandas as pd

from .evaluator import predict_graphs
from .metrics import regression_metrics, ranking_metrics
from ..data.dataset import SyntheticFEMDatasetGenerator
from ..data.graph_builder import build_graph
from ..utils.config import resolve_path
from ..utils.logging import write_json
from ..utils.visualization import plot_evaluation


def screen_designs(model, config, count=1000, top_k=10, output_dir="outputs/screening"):
    if count < 20 or not 1 <= top_k <= count:
        raise ValueError("Use count >= 20 and 1 <= top_k <= count")
    generator = SyntheticFEMDatasetGenerator(config)
    rng = np.random.default_rng(config.get("seed", 42) + 78901)
    meshes = [generator.sample_mesh(rng) for _ in range(count)]
    graphs = [build_graph(mesh, sample_id=f"candidate-{i}") for i, mesh in enumerate(meshes)]
    started = perf_counter()
    prediction, _ = predict_graphs(model, graphs, config["training"]["batch_size"])
    inference_seconds = perf_counter() - started
    selected = np.argsort(-prediction, kind="stable")[:top_k]
    started = perf_counter()
    solutions = {int(i): generator.solver.run(meshes[int(i)], rng) for i in selected}
    selected_solver_seconds = perf_counter() - started
    # Full-pool oracle evaluation is a demo audit, separate from Top-K validation.
    for i, mesh in enumerate(meshes):
        if i not in solutions:
            solutions[i] = generator.solver.run(mesh, rng)
    truth = np.array([solutions[i].stiffness for i in range(count)])
    best_pred, best_true = int(np.argmax(prediction)), int(np.argmax(truth))
    metrics = {**regression_metrics(truth, prediction), **ranking_metrics(truth, prediction)}
    output_dir = resolve_path(output_dir)
    result = {
        "provenance": "synthetic-only", "candidate_count": count, "selected_top_k": top_k,
        "metrics": metrics, "selected_indices": selected.tolist(),
        "selected_solver_calls": top_k, "additional_oracle_audit_calls": count - top_k,
        "inference_seconds": inference_seconds, "selected_solver_seconds": selected_solver_seconds,
        "best_predicted_design": {"index": best_pred, "parameters": meshes[best_pred].design_parameters,
                                  "predicted_stiffness": float(prediction[best_pred]),
                                  "true_stiffness": float(truth[best_pred])},
        "best_true_design": {"index": best_true, "parameters": meshes[best_true].design_parameters,
                             "true_stiffness": float(truth[best_true])},
        "disclaimer": "Full-pool labels are synthetic audit results, not an industrial speedup benchmark.",
    }
    write_json(output_dir / "screening.json", result)
    pd.DataFrame({"candidate": np.arange(count), "predicted_stiffness": prediction,
                  "true_stiffness": truth, "selected": np.isin(np.arange(count), selected)}).to_csv(
        output_dir / "candidates.csv", index=False
    )
    plot_evaluation(graphs, truth, prediction, None, output_dir / "figures")
    return result
