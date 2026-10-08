import torch
from torch import nn

from ..data.graph_builder import EDGE_FEATURES, NODE_FEATURES
from ..data.synthetic_mesh import DESIGN_KEYS


def mlp(input_dim, hidden_dim, output_dim, layer_norm=False):
    layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim)]
    if layer_norm:
        layers.append(nn.LayerNorm(output_dim))
    return nn.Sequential(*layers)


def graph_batch(data):
    batch = getattr(data, "batch", None)
    return batch if batch is not None else torch.zeros(
        data.num_nodes, dtype=torch.long, device=data.x.device
    )


class SurrogateBase(nn.Module):
    def __init__(self):
        super().__init__()
        for key, size in [("node", len(NODE_FEATURES)), ("edge", len(EDGE_FEATURES)),
                          ("design", len(DESIGN_KEYS))]:
            self.register_buffer(f"{key}_mean", torch.zeros(size))
            self.register_buffer(f"{key}_std", torch.ones(size))
        self.register_buffer("k_log_mean", torch.tensor(0.0))
        self.register_buffer("k_log_std", torch.tensor(1.0))
        self.register_buffer("u_scale", torch.tensor(1.0))

    def set_normalization(self, stats):
        for key, value in stats.items():
            getattr(self, key).copy_(value)

    def normalize_inputs(self, data):
        return ((data.x - self.node_mean) / self.node_std,
                (data.edge_attr - self.edge_mean) / self.edge_std)

    def decode(self, output):
        k = torch.exp((output["k_log"] * self.k_log_std + self.k_log_mean).clamp(-20, 30))
        u = output["u_norm"]
        return k, u * self.u_scale if u is not None else None
