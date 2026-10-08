import torch

from src.models import create_model
from src.training.trainer import fit_normalization


def test_meshgraphnet_permutation_equivariance(config, graph):
    torch.manual_seed(5)
    model = create_model("multitask", config)
    model.set_normalization(fit_normalization([graph]))
    model.eval()
    permutation = torch.randperm(graph.num_nodes)
    inverse = torch.argsort(permutation)
    changed = graph.clone()
    for attr in ("x", "pos", "boundary", "left_mask", "right_mask", "force", "y_u"):
        changed[attr] = graph[attr][permutation]
    changed.edge_index = inverse[graph.edge_index]
    original, shuffled = model(graph), model(changed)
    torch.testing.assert_close(original["k_log"], shuffled["k_log"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(original["u_norm"][permutation], shuffled["u_norm"], rtol=1e-4, atol=1e-5)
