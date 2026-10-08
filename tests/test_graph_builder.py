import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Batch

from src.data.graph_builder import build_graph
from src.data.synthetic_mesh import generate_mesh, sample_design


def test_graph_integrity(graph):
    assert graph.x.shape == (128, 18)
    assert graph.edge_attr.shape == (graph.edge_index.shape[1], 8)
    assert graph.y_u.shape == (128, 1)
    edges = set(map(tuple, graph.edge_index.T.tolist()))
    assert all((b, a) in edges and a != b for a, b in edges)
    assert len(edges) == graph.edge_index.size(1)
    adjacency = coo_matrix((np.ones(len(edges)), graph.edge_index.numpy()), shape=(128, 128))
    assert connected_components(adjacency, directed=False)[0] == 1
    assert torch.isfinite(graph.x).all() and torch.isfinite(graph.edge_attr).all()


def test_batch_has_no_cross_graph_edges_and_no_inference_labels(graph):
    batch = Batch.from_data_list([graph, graph.clone()])
    assert torch.equal(batch.batch[batch.edge_index[0]], batch.batch[batch.edge_index[1]])
    rng = np.random.default_rng(6)
    unlabeled = build_graph(generate_mesh(sample_design(rng), rng))
    assert "y_k" not in unlabeled and "y_u" not in unlabeled
