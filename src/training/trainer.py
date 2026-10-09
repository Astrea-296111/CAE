import hashlib
from collections import defaultdict
from time import perf_counter
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .evaluator import evaluate_model
from ..data.dataset import load_dataset
from ..data.contract import feature_schema, schema_hash, fit_training_domain
from ..losses.regression_loss import surrogate_loss
from ..models import create_model
from ..utils.config import resolve_path
from ..utils.logging import write_json
from ..utils.seed import set_seed


def fit_normalization(graphs):
    stats = {}
    for prefix, attr in [("node", "x"), ("edge", "edge_attr"), ("design", "design_parameters")]:
        values = torch.cat([getattr(g, attr) for g in graphs], dim=0)
        stats[f"{prefix}_mean"] = values.mean(0)
        stats[f"{prefix}_std"] = values.std(0, unbiased=False).clamp_min(1e-6)
    k = torch.cat([g.y_k for g in graphs]).log()
    u = torch.cat([g.y_u for g in graphs])
    stats.update(k_log_mean=k.mean(), k_log_std=k.std(unbiased=False).clamp_min(1e-6),
                 u_scale=u.square().mean().sqrt().clamp_min(1e-9))
    return stats


def train_model(kind, config, output_dir=None):
    set_seed(config.get("seed", 42), config["training"].get("threads", 2))
    settings = config["training"]
    device = settings.get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; set training.device=cpu")
    epochs = int(settings["epochs"])
    warmup = config["loss"].get("physics_warmup_epochs", 10)
    if epochs < 1 or (kind == "physics" and epochs <= warmup):
        raise ValueError("Physics runs need epochs > physics_warmup_epochs; all runs need epochs >= 1")
    dataset_path = resolve_path(config["dataset"]["path"])
    dataset = load_dataset(dataset_path)
    output_dir = resolve_path(output_dir or f"outputs/{kind}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(kind, config)
    model.set_normalization(fit_normalization(dataset["train"]))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    loader = DataLoader(dataset["train"], batch_size=settings["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=settings["batch_size"], shuffle=False)
    best, stale, history = float("inf"), 0, []
    checkpoint_path = output_dir / "best.pt"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    domain = fit_training_domain(dataset["train"])
    for epoch in range(epochs):
        started = perf_counter()
        model.train()
        train_total, seen = 0.0, 0
        part_totals = defaultdict(float)
        gradient_norm_max = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, parts = surrogate_loss(model(batch), batch, model, config, epoch, kind)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            gradient_norm_max = max(gradient_norm_max, float(gradient_norm))
            optimizer.step()
            train_total += loss.item() * batch.num_graphs
            seen += batch.num_graphs
            for key, value in parts.items():
                part_totals[key] += value * batch.num_graphs
        model.eval()
        val_total, val_seen = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # Select all models on supervised validation error only.
                loss, _ = surrogate_loss(model(batch), batch, model, config, epoch, "validation")
                val_total += loss.item() * batch.num_graphs
                val_seen += batch.num_graphs
        val_loss = val_total / val_seen
        if not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError(f"Nonfinite validation loss at epoch {epoch + 1}")
        eligible = kind != "physics" or epoch >= warmup
        if eligible and val_loss < best:
            best, stale = val_loss, 0
            torch.save({
                "state_dict": model.state_dict(), "config": config, "kind": kind,
                "epoch": epoch + 1, "validation_loss": best, "provenance": "synthetic-only",
                "dataset_sha256": dataset_hash,
                "checkpoint_version": 1, "feature_schema": feature_schema(),
                "feature_schema_sha256": schema_hash(), "training_domain": domain,
            }, checkpoint_path.with_suffix(".tmp"))
            checkpoint_path.with_suffix(".tmp").replace(checkpoint_path)
        elif eligible:
            stale += 1
        history.append({"epoch": epoch + 1, "train_loss": train_total / seen, "val_loss": val_loss,
                        "train_components": {key: value / seen for key, value in part_totals.items()},
                        "gradient_norm_max_before_clip": gradient_norm_max,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "seconds": perf_counter() - started})
        write_json(output_dir / "history.json", history)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"{kind}: epoch={epoch + 1} train={train_total / seen:.5f} val={val_loss:.5f}", flush=True)
        if stale >= settings.get("patience", epochs):
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    results = {split: evaluate_model(model, dataset[split], settings["batch_size"])[0]
               for split in ("val", "test", "ood")}
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "metrics.json", {
        "provenance": "synthetic-only", "kind": kind, "best_epoch": checkpoint["epoch"],
        "dataset_sha256": dataset_hash, "splits": results,
    })
    return model, results, checkpoint_path
