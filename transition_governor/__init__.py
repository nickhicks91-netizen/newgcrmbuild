"""
Transition Governor: Deterministic control kernel for AI system stability.

This package provides infrastructure for governing AI system transitions,
preventing instability, and ensuring graceful degradation under uncertainty.

Core principles:
- Transition governance is a precondition, not a feature
- No interpretation inside the governor
- All transitions are continuous and rate-limited
- Graceful degradation over correctness
"""

__version__ = "0.1.0"

from .core.governor import TransitionGovernor
from .core.state import AIState, GovernanceState
from .echozero_bridge.contract import GovernedContext

__all__ = [
    "TransitionGovernor",
    "AIState",
    "GovernanceState",
    "GovernedContext",
]
