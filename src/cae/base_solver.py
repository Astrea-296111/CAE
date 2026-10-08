from abc import ABC, abstractmethod


class BaseCAESolver(ABC):
    @abstractmethod
    def run(self, mesh):
        """Return stiffness, displacement, torque, angle and provenance."""
