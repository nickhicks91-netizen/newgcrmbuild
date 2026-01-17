"""
Core state representation for the Transition Governor.

This module defines the minimal state required for governance decisions.
No interpretation or semantic meaning is stored here.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GovernanceState(Enum):
    """
    Governor operating mode.

    NORMAL: System is stable, full capabilities available
    BROWNOUT: System is unstable, reduced capabilities enforced
    """
    NORMAL = "normal"
    BROWNOUT = "brownout"


class ToolState(Enum):
    """
    Tool execution state.

    These states are consumed by the governor but never interpreted.
    The governor only cares about stability signals, not tool semantics.
    """
    INACTIVE = "inactive"
    PENDING = "pending"
    ACTIVE = "active"
    ERROR = "error"


@dataclass
class AIState:
    """
    Complete state snapshot for governance decisions.

    This is the ONLY state the governor operates on. All values must be:
    - Measurable
    - Deterministic from inputs
    - Free of semantic interpretation

    Invariants:
    - All rates of change are computed from history
    - No state may be None after initialization
    - Agent ID is optional (single-agent mode)
    """

    # Entropy signals (primary stability indicators)
    entropy: float
    entropy_dot: float  # Rate of change

    # Confidence signals (inverse entropy proxy)
    confidence: float
    confidence_dot: float  # Rate of change

    # Tool state (external action stability)
    tool_state: ToolState

    # Resource usage
    context_length: int
    max_context_length: int

    # Fatigue accumulator (monotonic increasing)
    fatigue: float

    # Multi-agent support (optional)
    agent_id: Optional[str] = None

    # Previous state for rate computation (internal use only)
    _prev_entropy: Optional[float] = field(default=None, repr=False)
    _prev_confidence: Optional[float] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate invariants."""
        assert 0.0 <= self.entropy, "Entropy must be non-negative"
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
        assert 0 <= self.context_length <= self.max_context_length
        assert self.fatigue >= 0.0, "Fatigue must be non-negative"

    def update_rates(self, prev_state: Optional['AIState']) -> None:
        """
        Update rate of change from previous state.

        This is the ONLY way rates should be computed.
        Deterministic, no smoothing, no interpretation.
        """
        if prev_state is not None:
            self.entropy_dot = self.entropy - prev_state.entropy
            self.confidence_dot = self.confidence - prev_state.confidence
        else:
            # First step: no rate of change
            self.entropy_dot = 0.0
            self.confidence_dot = 0.0


@dataclass
class GovernorOutput:
    """
    Governor decision output.

    These are the ONLY controls the governor may exert.
    All values are limits, caps, or rate constraints.
    """

    # Governance mode
    governance_state: GovernanceState

    # Authority weights (sum to 1.0)
    authority_weights: dict[str, float]  # {reasoning, tools, memory, ...}

    # Confidence budget (cap on max confidence)
    confidence_cap: float

    # Rate limits
    max_tokens_per_step: Optional[int] = None
    max_tool_calls_per_step: int = 1

    # Transition metrics (for logging/debugging)
    transition_intensity: float = 0.0
    remaining_margin: float = 0.0

    def __post_init__(self):
        """Validate output invariants."""
        assert 0.0 <= self.confidence_cap <= 1.0
        total_authority = sum(self.authority_weights.values())
        assert abs(total_authority - 1.0) < 1e-6, "Authority weights must sum to 1.0"
        assert self.max_tool_calls_per_step >= 0
