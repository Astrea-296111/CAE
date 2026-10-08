import argparse

from _bootstrap import ROOT
from src.training.experiments import run_experiments, summarize_saved_experiments
from src.utils.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument("--report", default="experiments/results")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    runner = summarize_saved_experiments if args.summarize_only else run_experiments
    runner(load_config(args.config), args.output, args.report)
