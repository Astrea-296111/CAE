import argparse
import json

from .data.dataset import SyntheticFEMDatasetGenerator, load_dataset, save_dataset
from .data.candidates import generate_candidates, load_candidates
from .inference.pipeline import run_screening
from .inference.reliability import ReliabilityPolicy
from .training.baselines import train_baselines
from .training.benchmark import benchmark
from .training.evaluator import evaluate_model, load_checkpoint
from .training.trainer import train_model
from .utils.config import load_config, resolve_path
from .utils.logging import write_json
from .utils.seed import set_seed
from .utils.visualization import plot_dataset, plot_evaluation


def main(default_command=None):
    parser = argparse.ArgumentParser(description="Synthetic AI-CAE research pipeline")
    if default_command is None:
        parser.add_argument("command", choices=["generate", "candidates", "baseline", "train", "evaluate", "screen", "benchmark"])
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--model", choices=["mlp", "gcn", "meshgraphnet", "multitask", "physics"], default="meshgraphnet")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint", default="outputs/meshgraphnet/best.pt")
    parser.add_argument("--split", choices=["val", "test", "ood"], default="test")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--input", help="Candidate JSON manifest for screening")
    parser.add_argument("--policy", default="configs/screening.yaml")
    parser.add_argument("--candidate-seed", type=int, default=2026)
    parser.add_argument("--ood-fraction", type=float, default=0.2)
    parser.add_argument("--audit", action="store_true", help="Also solve the full pool for offline ranking metrics")
    args = parser.parse_args()
    command = default_command or args.command
    config = load_config(args.config)
    set_seed(config.get("seed", 42), config["training"].get("threads", 2))
    if command == "candidates":
        manifest = generate_candidates(config, args.count if args.count is not None else 100,
                                       args.candidate_seed, args.ood_fraction)
        path = resolve_path(args.output or "outputs/candidates.json")
        write_json(path, manifest)
        result = {"path": str(path), "count": len(manifest["candidates"])}
    elif command == "generate":
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
        model, checkpoint = load_checkpoint(resolve_path(args.checkpoint), require_contract=command == "screen")
        config = checkpoint["config"]
        set_seed(config.get("seed", 42), config["training"].get("threads", 2))
        if command == "evaluate":
            dataset = load_dataset(resolve_path(config["dataset"]["path"]))
            metrics, truth, prediction, displacement = evaluate_model(
                model, dataset[args.split], config["training"]["batch_size"]
            )
            directory = resolve_path(args.output or f"outputs/evaluation/{checkpoint['kind']}/{args.split}")
            write_json(directory / "evaluation_metrics.json", {"provenance": "synthetic-only", "split": args.split, "metrics": metrics})
            plot_evaluation(dataset[args.split], truth, prediction, displacement, directory / "figures")
            result = metrics
        elif command == "screen":
            import hashlib
            manifest = load_candidates(resolve_path(args.input)) if args.input else generate_candidates(
                config, args.count if args.count is not None else 1000, args.candidate_seed, args.ood_fraction)
            policy = ReliabilityPolicy(**load_config(args.policy))
            result = run_screening(model, checkpoint, manifest, resolve_path(args.output or "outputs/screening"),
                                   args.top_k, policy, audit=args.audit,
                                   checkpoint_sha256=hashlib.sha256(resolve_path(args.checkpoint).read_bytes()).hexdigest())
        elif command == "benchmark":
            result = benchmark(model, config, 100 if args.count is None else args.count,
                               args.repeats, args.output or "outputs/benchmark")
        else:
            raise ValueError(command)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
