from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path="configs/dataset.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    config = yaml.safe_load(path.read_text())
    if "base" in config:
        base = load_config(path.parent / config.pop("base"))
        config = merge_config(base, config)
    return config


def merge_config(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
