from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .metrics import regression_metrics, ranking_metrics
from ..losses.physics_consistency_loss import torsional_angle
from ..models import create_model
from ..data.contract import validate_checkpoint_contract


def load_checkpoint(path, device="cpu", require_contract=False):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("provenance") != "synthetic-only":
        raise ValueError("Expected a synthetic research checkpoint")
    validate_checkpoint_contract(checkpoint, require_contract)
    model = create_model(checkpoint["kind"], checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_graphs(model, graphs, batch_size=16):
    if not graphs:
        raise ValueError("Inference requires at least one graph")
    device = next(model.parameters()).device
    model.eval()
    predictions, displacements = [], []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        k, u = model.decode(model(batch))
        predictions.append(k.cpu().numpy())
        if u is not None:
            displacements.extend([v.cpu().numpy() for v in torch.split(u, batch.ptr.diff().tolist())])
    return np.concatenate(predictions), displacements or None


@torch.inference_mode()
def evaluate_model(model, graphs, batch_size=16):
    prediction, displacement = predict_graphs(model, graphs, batch_size)
    truth = np.array([g.y_k.item() for g in graphs])
    metrics = {**regression_metrics(truth, prediction), **ranking_metrics(truth, prediction)}
    if displacement is not None:
        true_u = np.concatenate([g.y_u.numpy() for g in graphs])
        pred_u = np.concatenate(displacement)
        metrics["node_displacement_MAE_m"] = float(np.abs(true_u - pred_u).mean())
        metrics["node_displacement_RMSE_m"] = float(np.sqrt(((true_u - pred_u) ** 2).mean()))
        angles = np.array([
            torsional_angle(torch.tensor(u), g).item() for g, u in zip(graphs, displacement)
        ])
        torques = np.array([g.torque.item() for g in graphs])
        metrics["nonpositive_angle_fraction"] = float(np.mean(angles <= 0))
        metrics["mean_torque_consistency_residual"] = float(
            np.mean(np.abs(prediction * angles / torques - 1))
        )
    return metrics, truth, prediction, displacement
