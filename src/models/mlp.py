from .common import SurrogateBase, mlp
from ..data.synthetic_mesh import DESIGN_KEYS


class DesignMLP(SurrogateBase):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.network = mlp(len(DESIGN_KEYS), hidden_dim, 1)

    def forward(self, data):
        inputs = (data.design_parameters - self.design_mean) / self.design_std
        return {"k_log": self.network(inputs).flatten(), "u_norm": None}
