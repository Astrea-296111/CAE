"""Train independently seeded MLP members on an existing fixed dataset."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import train_model
from src.utils.config import load_config, resolve_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", default="outputs/ensemble")
    args = parser.parse_args()
    base = load_config(args.config)
    for seed in range(base.get("seed", 42), base.get("seed", 42) + 3):
        config = deepcopy(base)
        config["seed"] = seed
        config["training"]["epochs"] = args.epochs
        train_model("mlp", config, resolve_path(args.output) / f"seed{seed}")
