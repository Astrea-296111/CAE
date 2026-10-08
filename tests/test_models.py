import pytest
import torch
from torch_geometric.data import Batch

from src.models import create_model
from src.training.trainer import fit_normalization


@pytest.mark.parametrize("kind", ["mlp", "gcn", "meshgraphnet", "multitask", "physics"])
def test_forward_backward_and_batch_independence(kind, config, graph):
    model = create_model(kind, config)
    model.set_normalization(fit_normalization([graph]))
    batch = Batch.from_data_list([graph, graph.clone()])
    model.eval()
    output = model(batch)
    assert output["k_log"].shape == (2,)
    k, u = model.decode(output)
    assert torch.isfinite(k).all() and (k > 0).all()
    single = model(graph)
    torch.testing.assert_close(single["k_log"][0], output["k_log"][0], rtol=1e-4, atol=1e-5)
    if u is not None:
        assert u.shape == (256, 1)
        assert torch.all(u[batch.boundary] == 0)
        torch.testing.assert_close(output["u_norm"][:128], single["u_norm"], rtol=1e-4, atol=1e-5)
    loss = output["k_log"].square().mean()
    if u is not None:
        loss = loss + output["u_norm"].square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
