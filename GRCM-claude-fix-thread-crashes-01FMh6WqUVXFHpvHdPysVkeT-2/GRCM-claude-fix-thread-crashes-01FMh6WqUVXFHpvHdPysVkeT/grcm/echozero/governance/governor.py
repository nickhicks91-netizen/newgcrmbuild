"""
Transition Governor: Core control kernel.

This is the main governor implementation. It consumes AIState and produces
GovernorOutput deterministically.

NO interpretation happens here. Only limits, caps, and rate constraints.
"""

import logging
from typing import Optional, List
from dataclasses import asdict

from .state import AIState, GovernorOutput, GovernanceState, ToolState
from .authority import AuthorityAllocator
from .degradation import (
    BrownoutController,
    BrownoutConfig,
    FatigueAccumulator,
    DegradationScheduler
)

# Configure logging for audit trail
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransitionGovernor:
    """
    Deterministic control kernel for AI system stability.

    The governor:
    1. Accepts AIState (measurements only)
    2. Computes transition intensity
    3. Evaluates stability margin
    4. Triggers brownout if margin exhausted
    5. Returns GovernorOutput (limits only)

    The governor NEVER:
    - Interprets semantic meaning
    - Evaluates truth or correctness
    - Makes decisions about content
    - Overrides based on "should" or "must"
    """

    # Transition intensity weights (tunable parameters)
    W_ENTROPY_DOT = 0.5
    W_CONFIDENCE_DOT = 0.3
    W_AUTHORITY_SHIFT = 0.2

    # Maximum allowed transition intensity before brownout
    MAX_MARGIN = 2.0

    def __init__(
        self,
        brownout_config: Optional[BrownoutConfig] = None,
        seed: int = 42,
        enable_logging: bool = True
    ):
        """
        Initialize governor.

        Args:
            brownout_config: Brownout configuration (uses defaults if None)
            seed: Random seed for deterministic behavior
            enable_logging: Whether to log state transitions
        """
        self.seed = seed
        self.enable_logging = enable_logging

        # Initialize subsystems
        self.authority_allocator = AuthorityAllocator(seed=seed)
        self.brownout_controller = BrownoutController(brownout_config)
        self.fatigue_accumulator = FatigueAccumulator(seed=seed)
        self.degradation_scheduler = DegradationScheduler()

        # State history for deterministic replay
        self.state_history: List[AIState] = []
        self.output_history: List[GovernorOutput] = []

        # Previous authority weights (for computing shift)
        self.prev_authority_weights = self.authority_allocator.DEFAULT_WEIGHTS.copy()

    def govern(self, state: AIState) -> GovernorOutput:
        """
        Main governance function.

        Args:
            state: Current AI system state

        Returns:
            Governor output (limits and caps)
        """
        # Update state rates if we have history
        if self.state_history:
            state.update_rates(self.state_history[-1])

        # Log input state
        if self.enable_logging:
            self._log_state("INPUT", state)

        # Step 1: Compute transition intensity
        transition_intensity = self._compute_transition_intensity(state)

        # Step 2: Compute stability margin
        remaining_margin = self.MAX_MARGIN - transition_intensity

        # Step 3: Accumulate fatigue
        tool_failed = (state.tool_state == ToolState.ERROR)
        fatigue = self.fatigue_accumulator.accumulate(
            transition_intensity,
            tool_failed=tool_failed
        )

        # Step 4: Evaluate brownout
        governance_state = self.brownout_controller.evaluate(remaining_margin)

        # Step 5: Reallocate authority
        authority_weights = self.authority_allocator.reallocate(
            transition_intensity,
            state.tool_state,
            in_brownout=(governance_state == GovernanceState.BROWNOUT)
        )

        # Step 6: Compute degradation factor
        degradation_factor = self.degradation_scheduler.compute_degradation_factor(
            fatigue,
            remaining_margin
        )

        # Step 7: Apply caps and limits
        base_confidence_cap = self.brownout_controller.get_confidence_cap()
        confidence_cap = self.degradation_scheduler.apply_degradation(
            base_confidence_cap,
            degradation_factor
        )

        max_tokens = self.brownout_controller.get_max_tokens()
        max_tool_calls = self.brownout_controller.get_max_tool_calls()

        # Step 8: Construct output
        output = GovernorOutput(
            governance_state=governance_state,
            authority_weights=authority_weights,
            confidence_cap=confidence_cap,
            max_tokens_per_step=max_tokens,
            max_tool_calls_per_step=max_tool_calls,
            transition_intensity=transition_intensity,
            remaining_margin=remaining_margin
        )

        # Update history
        self.state_history.append(state)
        self.output_history.append(output)
        self.prev_authority_weights = authority_weights.copy()

        # Log output
        if self.enable_logging:
            self._log_output("OUTPUT", output)

        return output

    def _compute_transition_intensity(self, state: AIState) -> float:
        """
        Compute transition intensity from state.

        Transition intensity = weighted sum of:
        - |entropy_dot| (rapid entropy changes)
        - |confidence_dot| (rapid confidence changes)
        - |authority_shift| (authority reallocation magnitude)

        Args:
            state: Current state

        Returns:
            Transition intensity (non-negative)
        """
        # Component 1: Entropy rate of change
        entropy_component = self.W_ENTROPY_DOT * abs(state.entropy_dot)

        # Component 2: Confidence rate of change
        confidence_component = self.W_CONFIDENCE_DOT * abs(state.confidence_dot)

        # Component 3: Authority shift (from previous weights)
        current_weights = self.authority_allocator.current_weights
        authority_shift = sum(
            abs(current_weights[k] - self.prev_authority_weights[k])
            for k in current_weights.keys()
        )
        authority_component = self.W_AUTHORITY_SHIFT * authority_shift

        # Total intensity
        intensity = entropy_component + confidence_component + authority_component

        return float(intensity)

    def _log_state(self, label: str, state: AIState) -> None:
        """
        Log state for debugging and audit.

        Args:
            label: Log label
            state: State to log
        """
        logger.info(f"[{label}] {asdict(state)}")

    def _log_output(self, label: str, output: GovernorOutput) -> None:
        """
        Log output for debugging and audit.

        Args:
            label: Log label
            output: Output to log
        """
        logger.info(f"[{label}] {asdict(output)}")

    def reset(self) -> None:
        """
        Reset governor to initial state.

        Use this for:
        - Starting new sessions
        - Testing deterministic replay
        - Clearing accumulated fatigue
        """
        self.authority_allocator.reset()
        self.brownout_controller.reset()
        self.fatigue_accumulator.reset()
        self.state_history.clear()
        self.output_history.clear()
        self.prev_authority_weights = self.authority_allocator.DEFAULT_WEIGHTS.copy()

        if self.enable_logging:
            logger.info("[RESET] Governor reset to initial state")

    def get_state_history(self) -> List[AIState]:
        """
        Get full state history for deterministic replay.

        Returns:
            List of all input states
        """
        return self.state_history.copy()

    def get_output_history(self) -> List[GovernorOutput]:
        """
        Get full output history for deterministic replay.

        Returns:
            List of all outputs
        """
        return self.output_history.copy()

    def replay_from_history(self, states: List[AIState]) -> List[GovernorOutput]:
        """
        Replay governance decisions from state history.

        This MUST produce identical outputs given identical inputs.
        If it doesn't, the governor is not deterministic.

        Args:
            states: List of states to replay

        Returns:
            List of outputs (should match original if deterministic)
        """
        self.reset()
        outputs = []

        for state in states:
            output = self.govern(state)
            outputs.append(output)

        return outputs
