import torch

from ..models.common import graph_batch


def torsional_angle(displacement, data):
    batch = graph_batch(data)
    n_graphs = data.torque.numel()
    values = displacement.flatten()

    def average(mask):
        total = values.new_zeros(n_graphs).index_add(0, batch[mask], values[mask])
        count = values.new_zeros(n_graphs).index_add(
            0, batch[mask], torch.ones_like(values[mask])
        )
        if torch.any(count == 0):
            raise ValueError("Each graph requires left and right load groups")
        return total / count

    return (average(data.left_mask) - average(data.right_mask)) / data.span


def physics_consistency_loss(stiffness, displacement, data, stiffness_scale):
    angle = torsional_angle(displacement, data)
    # Bound the inverse during early training; retain a signed-angle penalty.
    angle_floor = data.torque / (20 * stiffness_scale)
    safe_angle = torch.maximum(angle, angle_floor)
    stiffness_physics = data.torque / safe_angle
    mismatch = ((stiffness - stiffness_physics) / stiffness_scale).square().mean()
    sign_penalty = torch.relu(-angle / angle_floor).square().mean()
    return mismatch + sign_penalty
