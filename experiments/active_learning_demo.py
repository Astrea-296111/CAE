"""Optional ensemble acquisition; query labels only after uncertainty ranking."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SyntheticFEMDatasetGenerator, load_dataset, save_dataset
from src.data.graph_builder import build_graph
from src.training.evaluator import load_checkpoint, predict_graphs
from src.utils.config import resolve_path
from src.utils.logging import write_json
from src.utils.seed import set_seed


def acquire(checkpoints, count=100, queries=10, output="outputs/active_learning"):
    if len(checkpoints) < 2 or not 1 <= queries <= count:
        raise ValueError("Use at least two ensemble checkpoints and 1 <= queries <= count")
    members = [load_checkpoint(resolve_path(path)) for path in checkpoints]
    config = members[0][1]["config"]
    if len({m[1]["dataset_sha256"] for m in members}) != 1:
        raise ValueError("Ensemble members must share the same training dataset")
    if len({m[1]["kind"] for m in members}) != 1:
        raise ValueError("Use independently seeded models of the same architecture")
    if len({m[1]["config"].get("seed", 42) for m in members}) != len(members):
        raise ValueError("Ensemble members need different training seeds")
    set_seed(config.get("seed", 42), config["training"].get("threads", 2))
    generator = SyntheticFEMDatasetGenerator(config)
    rng = np.random.default_rng(config.get("seed", 42) + 99123)
    meshes = [generator.sample_mesh(rng) for _ in range(count)]
    graphs = [build_graph(m, sample_id=f"active-{config.get('seed', 42)}-{i}") for i, m in enumerate(meshes)]
    predictions = np.stack([predict_graphs(model, graphs)[0] for model, _ in members])
    variance = predictions.var(axis=0)
    selected = np.argsort(-variance, kind="stable")[:queries]
    queried = [build_graph(meshes[int(i)], generator.solver.run(meshes[int(i)], rng),
                           graphs[int(i)].sample_id) for i in selected]
    dataset = load_dataset(resolve_path(config["dataset"]["path"]))
    existing = {g.sample_id for split in ("train", "val", "test", "ood") for g in dataset[split]}
    if any(g.sample_id in existing for g in queried):
        raise ValueError("This acquisition pool was already queried")
    dataset["train"] = dataset["train"] + queried
    dataset["metadata"]["active_learning"] = {"queries": queries, "pool_size": count}
    output = resolve_path(output)
    save_dataset(dataset, output / "augmented_dataset.pt")
    result = {"provenance": "synthetic-only", "selected": selected.tolist(),
              "prediction_variance": variance[selected].tolist(), "solver_calls": queries,
              "next_step": "Retrain with dataset.path set to augmented_dataset.pt; test splits remain fixed.",
              "limitation": "Ensemble variance is an acquisition score, not a calibrated confidence interval."}
    write_json(output / "acquisition.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--output", default="outputs/active_learning")
    args = parser.parse_args()
    print(acquire(args.checkpoint, args.count, args.queries, args.output))
