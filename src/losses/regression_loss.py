import torch
from torch.nn.functional import mse_loss

from .physics_consistency_loss import physics_consistency_loss


def surrogate_loss(output, data, model, config, epoch=0, kind="meshgraphnet"):
    settings = config["loss"]
    target_k = (data.y_k.log() - model.k_log_mean) / model.k_log_std
    stiffness_loss = mse_loss(output["k_log"], target_k)
    total = settings.get("lambda_stiffness", 1) * stiffness_loss
    parts = {"stiffness": stiffness_loss.detach().item()}
    if output["u_norm"] is not None:
        # Average per graph so large meshes cannot dominate multi-task training.
        squared_error = (output["u_norm"] - data.y_u / model.u_scale).square().mean(-1)
        batch = data.batch
        sums = squared_error.new_zeros(data.num_graphs).index_add(0, batch, squared_error)
        count = torch.bincount(batch, minlength=data.num_graphs).to(sums.dtype)
        displacement_loss = (sums / count).mean()
        total = total + settings.get("lambda_displacement", 1) * displacement_loss
        parts["displacement"] = displacement_loss.detach().item()
        warmup = settings.get("physics_warmup_epochs", 10)
        weight = settings.get("lambda_physics", 0.1) if kind == "physics" else 0.0
        if weight and epoch >= warmup:
            k, u = model.decode(output)
            physics = physics_consistency_loss(k, u, data, model.k_log_mean.exp())
            total = total + weight * physics
            parts["physics"] = physics.detach().item()
    return total, parts
