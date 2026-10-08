from .meshgraphnet import MeshGraphNet


class MultiTaskMeshGraphNet(MeshGraphNet):
    def __init__(self, hidden_dim=128, message_passing_steps=6):
        super().__init__(hidden_dim, message_passing_steps, multitask=True)
