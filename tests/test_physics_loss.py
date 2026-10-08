import pytest
import torch
from torch_geometric.data import Batch

from src.losses.physics_consistency_loss import physics_consistency_loss, torsional_angle


def test_true_field_consistency_and_finite_zero_field(graph):
    batch = Batch.from_data_list([graph, graph.clone()])
    loss = physics_consistency_loss(batch.y_k, batch.y_u, batch, graph.y_k.mean())
    assert loss.item() < 1e-10
    angle = torsional_angle(batch.y_u, batch)
    torch.testing.assert_close(batch.y_k * angle, batch.torque, rtol=1e-5, atol=1e-3)
    zero = torch.zeros_like(batch.y_u, requires_grad=True)
    finite = physics_consistency_loss(batch.y_k, zero, batch, graph.y_k.mean())
    assert torch.isfinite(finite)
    finite.backward()
    assert torch.isfinite(zero.grad).all()
    wrong = physics_consistency_loss(batch.y_k, -batch.y_u, batch, graph.y_k.mean())
    assert wrong > loss


def test_missing_load_group_fails(graph):
    graph.left_mask[:] = False
    with pytest.raises(ValueError):
        torsional_angle(graph.y_u, graph)
