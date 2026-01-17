"""
Brownout and degradation logic.

This module implements graceful degradation under instability.
Brownout is a SUCCESS state - it prevents hallucination and unsafe behavior.

Core principle:
- Reduced capability is preferred to fabricated output
- Brownout is entered when stability margin is exhausted
- Recovery is smooth and hysteresis-based
"""

from dataclasses import dataclass
from typing import Optional
from .state import GovernanceState


@dataclass
class BrownoutConfig:
    """
    Configuration for brownout behavior.

    These parameters control when brownout is triggered and how aggressive
    the degradation is.
    """

    # Margin thresholds
    brownout_entry_threshold: float = 0.0  # Enter brownout when margin < this
    brownout_exit_threshold: float = 0.5   # Exit brownout when margin > this (hysteresis)

    # Confidence caps
    normal_confidence_cap: float = 1.0
    brownout_confidence_cap: float = 0.3  # Aggressive cap in brownout

    # Rate limits
    normal_max_tokens: Optional[int] = None  # No limit in normal
    brownout_max_tokens: int = 50           # Terse output in brownout

    # Tool limits
    normal_max_tool_calls: int = 5
    brownout_max_tool_calls: int = 0  # Zero tools in brownout

    def __post_init__(self):
        """Validate config."""
        assert self.brownout_exit_threshold > self.brownout_entry_threshold, \
            "Exit threshold must be higher than entry (hysteresis)"
        assert 0.0 <= self.brownout_confidence_cap <= self.normal_confidence_cap <= 1.0
        assert self.brownout_max_tool_calls == 0, "Tools must be disabled in brownout"


class BrownoutController:
    """
    Manages brownout state transitions.

    Implements hysteresis to prevent oscillation:
    - Entry threshold < Exit threshold
    - State is sticky (requires clear signal to change)

    Invariants:
    - State can only be NORMAL or BROWNOUT
    - Transitions are logged for debugging
    """

    def __init__(self, config: Optional[BrownoutConfig] = None):
        """
        Initialize controller.

        Args:
            config: Brownout configuration (uses defaults if None)
        """
        self.config = config or BrownoutConfig()
        self.state = GovernanceState.NORMAL
        self.transition_count = 0

    def evaluate(self, remaining_margin: float) -> GovernanceState:
        """
        Evaluate whether to enter/exit brownout.

        Args:
            remaining_margin: Stability margin (positive = stable, negative = unstable)

        Returns:
            New governance state
        """
        if self.state == GovernanceState.NORMAL:
            # Check for brownout entry
            if remaining_margin < self.config.brownout_entry_threshold:
                self._transition_to(GovernanceState.BROWNOUT)
        else:  # BROWNOUT
            # Check for brownout exit (with hysteresis)
            if remaining_margin > self.config.brownout_exit_threshold:
                self._transition_to(GovernanceState.NORMAL)

        return self.state

    def _transition_to(self, new_state: GovernanceState) -> None:
        """
        Transition to new state.

        Args:
            new_state: Target state
        """
        if new_state != self.state:
            self.transition_count += 1
            self.state = new_state

    def get_confidence_cap(self) -> float:
        """
        Get current confidence cap based on state.

        Returns:
            Maximum allowed confidence
        """
        if self.state == GovernanceState.BROWNOUT:
            return self.config.brownout_confidence_cap
        return self.config.normal_confidence_cap

    def get_max_tokens(self) -> Optional[int]:
        """
        Get maximum tokens per step.

        Returns:
            Token limit (None = no limit)
        """
        if self.state == GovernanceState.BROWNOUT:
            return self.config.brownout_max_tokens
        return self.config.normal_max_tokens

    def get_max_tool_calls(self) -> int:
        """
        Get maximum tool calls per step.

        Returns:
            Tool call limit
        """
        if self.state == GovernanceState.BROWNOUT:
            return self.config.brownout_max_tool_calls
        return self.config.normal_max_tool_calls

    def reset(self) -> None:
        """Reset to normal state."""
        self.state = GovernanceState.NORMAL
        self.transition_count = 0


class FatigueAccumulator:
    """
    Monotonic fatigue tracking.

    Fatigue increases with:
    - Time (monotonic)
    - High transition intensity (accelerated)
    - Tool failures (penalty)

    Fatigue NEVER decreases (until explicit reset).
    This prevents the system from "forgetting" past instability.
    """

    def __init__(self, base_rate: float = 0.01, seed: int = 42):
        """
        Initialize accumulator.

        Args:
            base_rate: Baseline fatigue accumulation per step
            seed: Random seed for determinism
        """
        self.base_rate = base_rate
        self.seed = seed
        self.fatigue = 0.0

    def accumulate(
        self,
        transition_intensity: float,
        tool_failed: bool = False,
        dt: float = 1.0
    ) -> float:
        """
        Accumulate fatigue.

        Args:
            transition_intensity: Current transition magnitude
            tool_failed: Whether a tool failed this step
            dt: Time delta (for rate scaling)

        Returns:
            Updated fatigue value
        """
        # Base accumulation
        delta = self.base_rate * dt

        # Accelerate under high transition intensity
        delta += transition_intensity * 0.05 * dt

        # Penalty for tool failure
        if tool_failed:
            delta += 1.0

        self.fatigue += delta
        return self.fatigue

    def get_fatigue(self) -> float:
        """Get current fatigue level."""
        return self.fatigue

    def reset(self) -> None:
        """Reset fatigue (use sparingly)."""
        self.fatigue = 0.0


class DegradationScheduler:
    """
    Computes degradation level from fatigue and margin.

    Degradation affects:
    - Confidence cap (reduced)
    - Authority allocation (more conservative)
    - Rate limits (more restrictive)

    This is separate from brownout - degradation is gradual, brownout is binary.
    """

    def __init__(self, fatigue_threshold: float = 10.0):
        """
        Initialize scheduler.

        Args:
            fatigue_threshold: Fatigue level at which max degradation occurs
        """
        self.fatigue_threshold = fatigue_threshold

    def compute_degradation_factor(self, fatigue: float, margin: float) -> float:
        """
        Compute current degradation factor.

        Args:
            fatigue: Current fatigue level
            margin: Current stability margin

        Returns:
            Degradation factor in [0, 1] (0 = full degradation, 1 = no degradation)
        """
        # Fatigue-based degradation (gradual)
        fatigue_factor = max(0.0, 1.0 - fatigue / self.fatigue_threshold)

        # Margin-based degradation (immediate response to instability)
        margin_factor = max(0.0, min(1.0, margin))

        # Combined (take minimum = most conservative)
        degradation_factor = min(fatigue_factor, margin_factor)

        return degradation_factor

    def apply_degradation(
        self,
        base_confidence_cap: float,
        degradation_factor: float
    ) -> float:
        """
        Apply degradation to confidence cap.

        Args:
            base_confidence_cap: Starting confidence cap
            degradation_factor: Degradation factor in [0, 1]

        Returns:
            Degraded confidence cap
        """
        # Minimum cap even under full degradation
        min_cap = 0.1

        degraded_cap = base_confidence_cap * degradation_factor
        return max(min_cap, degraded_cap)
