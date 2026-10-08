import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from .common import SurrogateBase, graph_batch, mlp
from ..data.graph_builder import NODE_FEATURES, EDGE_FEATURES


class ProcessorBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_update = mlp(3 * hidden_dim, hidden_dim, hidden_dim, layer_norm=True)
        self.node_update = mlp(2 * hidden_dim, hidden_dim, hidden_dim, layer_norm=True)

    def forward(self, nodes, edges, edge_index):
        source, target = edge_index
        edges = edges + self.edge_update(torch.cat([edges, nodes[source], nodes[target]], -1))
        aggregate = torch.zeros_like(nodes).index_add(0, target, edges)
        degree = torch.bincount(target, minlength=nodes.size(0)).clamp_min(1)
        aggregate = aggregate / degree[:, None]
        nodes = nodes + self.node_update(torch.cat([nodes, aggregate], -1))
        return nodes, edges


class MeshGraphNet(SurrogateBase):
    def __init__(self, hidden_dim=128, message_passing_steps=6, multitask=False):
        super().__init__()
        self.node_encoder = mlp(len(NODE_FEATURES), hidden_dim, hidden_dim, True)
        self.edge_encoder = mlp(len(EDGE_FEATURES), hidden_dim, hidden_dim, True)
        self.processor = nn.ModuleList([ProcessorBlock(hidden_dim)
                                        for _ in range(message_passing_steps)])
        self.stiffness_head = mlp(2 * hidden_dim, hidden_dim, 1)
        self.node_decoder = mlp(hidden_dim, hidden_dim, 1) if multitask else None

    def forward(self, data):
        x, edge_attr = self.normalize_inputs(data)
        nodes, edges = self.node_encoder(x), self.edge_encoder(edge_attr)
        for block in self.processor:
            nodes, edges = block(nodes, edges, data.edge_index)
        batch = graph_batch(data)
        pooled = torch.cat([global_mean_pool(nodes, batch), global_max_pool(nodes, batch)], -1)
        u = self.node_decoder(nodes) if self.node_decoder is not None else None
        if u is not None:
            u = u * (~data.boundary).to(u.dtype)[:, None]
        return {"k_log": self.stiffness_head(pooled).flatten(), "u_norm": u}
