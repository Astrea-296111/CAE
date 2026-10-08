from .base_solver import BaseCAESolver


class OptiStructAdapter(BaseCAESolver):
    """Extension point only: no solver invocation, license or real CAE data."""

    def run(self, mesh):
        raise NotImplementedError(
            "Requires an authorized exporter, solver runner and result parser. "
            "Map node IDs, boundary conditions and SI units explicitly; see "
            "docs/REAL_CAE_MIGRATION_GUIDE_CN.md."
        )
