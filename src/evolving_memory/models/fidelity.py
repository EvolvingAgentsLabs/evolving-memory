"""Fidelity weights — epistemological trust per trace source.

Real-world experiences are more reliable than simulated or dreamed ones.
These weights multiply into the dream engine's consolidation scoring.
"""

from .hierarchy import TraceSource

FIDELITY_WEIGHTS: dict[TraceSource, float] = {
    TraceSource.REAL_WORLD: 1.0,
    TraceSource.SIM_3D: 0.8,
    TraceSource.SIM_2D: 0.5,
    TraceSource.DREAM_TEXT: 0.3,
    TraceSource.AGENT: 0.7,
    TraceSource.UNKNOWN_SOURCE: 0.5,
}


def get_fidelity_weight(source: TraceSource) -> float:
    """Return the fidelity weight for a given trace source."""
    return FIDELITY_WEIGHTS.get(source, 0.5)
