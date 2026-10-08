import argparse
import json

from .data.dataset import SyntheticFEMDatasetGenerator, load_dataset, save_dataset
from .training.baselines import train_baselines
from .training.benchmark import benchmark
from .training.evaluator import evaluate_model, load_checkpoint
from .training.screening import screen_designs
from .training.trainer import train_model
from .utils.config import load_config, resolve_path
from .utils.logging import write_json
from .utils.seed import set_seed
from .utils.visualization import plot_dataset, plot_evaluation


def main(default_command=None):
    parser = argparse.ArgumentParser(description="Synthetic AI-CAE research pipeline")
    if default_command is None:
        parser.add_argument("command", choices=["generate", "baseline", "train", "evaluate", "screen", "benchmark"])
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--model", choices=["mlp", "gcn", "meshgraphnet", "multitask", "physics"], default="meshgraphnet")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint", default="outputs/meshgraphnet/best.pt")
    parser.add_argument("--split", choices=["val", "test", "ood"], default="test")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    command = default_command or args.command
    config = load_config(args.config)
    set_seed(config.get("seed", 42), config["training"].get("threads", 2))
    if command == "generate":
        dataset = SyntheticFEMDatasetGenerator(config).generate()
        path = resolve_path(args.output or config["dataset"]["path"])
        save_dataset(dataset, path)
        plot_dataset(dataset, resolve_path("outputs/dataset/figures"))
        result = {"path": str(path), "sizes": {s: len(dataset[s]) for s in ("train", "val", "test", "ood")}}
    elif command == "baseline":
        result = train_baselines(config, args.output or "outputs/baselines")
    elif command in {"train", "gcn", "meshgraphnet", "multitask"}:
        kind = {"gcn": "gcn", "meshgraphnet": "meshgraphnet", "multitask": "physics"}.get(command, args.model)
        _, metrics, path = train_model(kind, config, args.output)
        result = {"checkpoint": str(path), "splits": metrics}
    else:
        model, checkpoint = load_checkpoint(resolve_path(args.checkpoint))
        config = checkpoint["config"]
        set_seed(config.get("seed", 42), config["training"].get("threads", 2))
        if command == "evaluate":
            dataset = load_dataset(resolve_path(config["dataset"]["path"]))
            metrics, truth, prediction, displacement = evaluate_model(
                model, dataset[args.split], config["training"]["batch_size"]
            )
            directory = resolve_path(args.output or f"outputs/evaluation/{checkpoint['kind']}/{args.split}")
            write_json(directory / "metrics.json", {"provenance": "synthetic-only", "split": args.split, "metrics": metrics})
            plot_evaluation(dataset[args.split], truth, prediction, displacement, directory / "figures")
            result = metrics
        elif command == "screen":
            result = screen_designs(model, config, args.count or 1000, args.top_k, args.output or "outputs/screening")
        elif command == "benchmark":
            result = benchmark(model, config, args.count or 100, args.repeats, args.output or "outputs/benchmark")
        else:
            raise ValueError(command)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
