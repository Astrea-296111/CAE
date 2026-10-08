from .base_solver import BaseCAESolver
from ..data.synthetic_solver import SyntheticPhysicsSolver


class SyntheticCAESolver(BaseCAESolver):
    def __init__(self):
        self.solver = SyntheticPhysicsSolver()

    def run(self, mesh):
        return self.solver.run(mesh)
