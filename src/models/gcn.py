import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from .common import SurrogateBase, graph_batch, mlp
from ..data.graph_builder import NODE_FEATURES


class GCNSurrogate(SurrogateBase):
    def __init__(self, hidden_dim=128, message_passing_steps=6):
        super().__init__()
        self.input = nn.Linear(len(NODE_FEATURES), hidden_dim)
        self.layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim)
                                     for _ in range(message_passing_steps)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                    for _ in range(message_passing_steps)])
        self.head = mlp(2 * hidden_dim, hidden_dim, 1)

    def forward(self, data):
        x, _ = self.normalize_inputs(data)
        h = self.input(x)
        for conv, norm in zip(self.layers, self.norms):
            h = norm(h + torch.nn.functional.silu(
                conv(h, data.edge_index, edge_weight=data.edge_attr[:, -1])
            ))
        batch = graph_batch(data)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        return {"k_log": self.head(pooled).flatten(), "u_norm": None}
