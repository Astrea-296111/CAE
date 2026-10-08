from .gcn import GCNSurrogate
from .meshgraphnet import MeshGraphNet
from .mlp import DesignMLP
from .multitask_model import MultiTaskMeshGraphNet


def create_model(kind, config):
    settings = config["model"]
    hidden = int(settings.get("hidden_dim", 128))
    steps = int(settings.get("message_passing_steps", 6))
    if hidden < 4 or steps < 1:
        raise ValueError("hidden_dim >= 4 and message_passing_steps >= 1 required")
    if kind == "mlp":
        return DesignMLP(hidden)
    if kind == "gcn":
        return GCNSurrogate(hidden, steps)
    if kind == "meshgraphnet":
        return MeshGraphNet(hidden, steps)
    if kind in {"multitask", "physics"}:
        return MultiTaskMeshGraphNet(hidden, steps)
    raise ValueError(f"Unknown model: {kind}")
