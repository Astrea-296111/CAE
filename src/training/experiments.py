from importlib.metadata import version
from pathlib import Path
import json
import shutil

import pandas as pd
import torch

from .baselines import train_baselines
from .benchmark import benchmark
from .evaluator import evaluate_model
from .screening import screen_designs
from .trainer import train_model
from ..data.dataset import SyntheticFEMDatasetGenerator, save_dataset
from ..utils.config import resolve_path
from ..utils.logging import write_json
from ..utils.visualization import plot_dataset, plot_evaluation


def write_experiment_summary(config, results, report):
    report = resolve_path(report)
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, metrics in results.items():
        for split, values in metrics.items():
            rows.append({"model": name, "split": split, **values})
    pd.DataFrame(rows).to_csv(report / "comparison.csv", index=False)
    write_json(report / "metrics.json", {"provenance": "synthetic-only", "config": config, "models": results})


def summarize_saved_experiments(config, output, report):
    output = resolve_path(output)
    results = json.loads((output / "baselines/metrics.json").read_text())["models"]
    for kind in ("gcn", "meshgraphnet", "multitask", "physics"):
        results[kind] = json.loads((output / kind / "metrics.json").read_text())["splits"]
    write_experiment_summary(config, results, report)
    return results


def run_experiments(config, output="outputs/demo", report="experiments/results"):
    output, report = resolve_path(output), resolve_path(report)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    dataset = SyntheticFEMDatasetGenerator(config).generate()
    save_dataset(dataset, resolve_path(config["dataset"]["path"]))
    plot_dataset(dataset, report / "figures")
    baselines = train_baselines(config, output / "baselines")
    results = dict(baselines)
    final_model = None
    for kind in ("gcn", "meshgraphnet", "multitask", "physics"):
        model, metrics, checkpoint_path = train_model(kind, config, output / kind)
        results[kind] = metrics
        if kind == "physics":
            final_model = model
            metrics, truth, prediction, displacement = evaluate_model(model, dataset["test"])
            plot_evaluation(dataset["test"], truth, prediction, displacement, report / "figures")
            shutil.copy2(checkpoint_path, report / "demo_physics.pt")
    write_experiment_summary(config, results, report)
    screen_designs(final_model, config, count=1000, top_k=10, output_dir=report / "screening")
    benchmark(final_model, config, count=100, repeats=5, output_dir=report / "benchmark")
    write_json(report / "environment.json", {
        "provenance": "synthetic-only", "packages": {name: version(name) for name in (
            "torch", "torch-geometric", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "PyYAML"
        )}, "torch_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
        "split_sizes": {s: len(dataset[s]) for s in ("train", "val", "test", "ood")},
    })
    return results
