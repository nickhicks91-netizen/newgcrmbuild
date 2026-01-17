"""
Test core governor functionality.

Tests:
- Deterministic behavior
- State tracking
- Authority redistribution
- Transition intensity computation
"""

import pytest
import numpy as np

from transition_governor.core.governor import TransitionGovernor
from transition_governor.core.state import AIState, ToolState, GovernanceState
from transition_governor.core.degradation import BrownoutConfig


class TestGovernorDeterminism:
    """Test governor deterministic behavior."""

    def test_deterministic_output(self):
        """Governor must produce identical outputs for identical inputs."""
        seed = 42
        gov1 = TransitionGovernor(seed=seed, enable_logging=False)
        gov2 = TransitionGovernor(seed=seed, enable_logging=False)

        # Create identical states
        state1 = AIState(
            entropy=1.5,
            entropy_dot=0.1,
            confidence=0.7,
            confidence_dot=-0.05,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.5
        )

        state2 = AIState(
            entropy=1.5,
            entropy_dot=0.1,
            confidence=0.7,
            confidence_dot=-0.05,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.5
        )

        # Process states
        output1 = gov1.govern(state1)
        output2 = gov2.govern(state2)

        # Outputs must be identical
        assert output1.governance_state == output2.governance_state
        assert output1.confidence_cap == output2.confidence_cap
        assert output1.authority_weights == output2.authority_weights
        assert output1.transition_intensity == output2.transition_intensity

    def test_deterministic_replay(self):
        """Replay from history must produce identical results."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Create sequence of states
        states = []
        for i in range(10):
            state = AIState(
                entropy=1.0 + i * 0.1,
                entropy_dot=0.0,
                confidence=0.8 - i * 0.05,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=100 + i * 10,
                max_context_length=2048,
                fatigue=i * 0.1
            )
            states.append(state)

        # Process states
        outputs1 = [gov.govern(state) for state in states]

        # Replay from history
        outputs2 = gov.replay_from_history(states)

        # Must be identical
        assert len(outputs1) == len(outputs2)
        for o1, o2 in zip(outputs1, outputs2):
            assert o1.governance_state == o2.governance_state
            assert o1.transition_intensity == o2.transition_intensity
            assert o1.confidence_cap == o2.confidence_cap


class TestTransitionIntensity:
    """Test transition intensity computation."""

    def test_intensity_increases_with_entropy_change(self):
        """High entropy change should increase transition intensity."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Low entropy change
        state_low = AIState(
            entropy=1.0,
            entropy_dot=0.1,  # Small change
            confidence=0.8,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        # High entropy change
        state_high = AIState(
            entropy=1.0,
            entropy_dot=2.0,  # Large change
            confidence=0.8,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output_low = gov.govern(state_low)
        gov.reset()
        output_high = gov.govern(state_high)

        assert output_high.transition_intensity > output_low.transition_intensity

    def test_intensity_increases_with_confidence_change(self):
        """High confidence change should increase transition intensity."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Low confidence change
        state_low = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.8,
            confidence_dot=0.05,  # Small change
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        # High confidence change
        state_high = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.8,
            confidence_dot=0.5,  # Large change
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output_low = gov.govern(state_low)
        gov.reset()
        output_high = gov.govern(state_high)

        assert output_high.transition_intensity > output_low.transition_intensity


class TestAuthorityRedistribution:
    """Test authority weight reallocation."""

    def test_tool_authority_reduced_on_error(self):
        """Tool authority should decrease when tool fails."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Normal state
        state_normal = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.8,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        # Tool error state
        state_error = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.8,
            confidence_dot=0.0,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output_normal = gov.govern(state_normal)
        gov.reset()
        output_error = gov.govern(state_error)

        # Tool authority should be lower when tool failed
        assert output_error.authority_weights["tools"] < output_normal.authority_weights["tools"]

    def test_authority_sums_to_one(self):
        """Authority weights must always sum to 1.0."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        for i in range(20):
            state = AIState(
                entropy=np.random.rand() * 3.0,
                entropy_dot=np.random.randn() * 0.5,
                confidence=np.random.rand(),
                confidence_dot=np.random.randn() * 0.2,
                tool_state=np.random.choice(list(ToolState)),
                context_length=int(np.random.rand() * 2000),
                max_context_length=2048,
                fatigue=np.random.rand() * 5.0
            )

            output = gov.govern(state)

            # Check sum
            total = sum(output.authority_weights.values())
            assert abs(total - 1.0) < 1e-6, f"Authority weights sum to {total}, not 1.0"


class TestFatigueAccumulation:
    """Test fatigue tracking."""

    def test_fatigue_increases_monotonically(self):
        """Fatigue should never decrease without reset."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        prev_fatigue = 0.0

        for i in range(20):
            state = AIState(
                entropy=1.0 + i * 0.1,
                entropy_dot=0.1,
                confidence=0.8 - i * 0.02,
                confidence_dot=-0.01,
                tool_state=ToolState.INACTIVE,
                context_length=100,
                max_context_length=2048,
                fatigue=0.0  # Will be updated by governor
            )

            output = gov.govern(state)
            current_fatigue = gov.fatigue_accumulator.get_fatigue()

            # Fatigue must increase or stay the same
            assert current_fatigue >= prev_fatigue
            prev_fatigue = current_fatigue

    def test_tool_failure_increases_fatigue(self):
        """Tool failures should accelerate fatigue."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # State without tool failure
        for i in range(5):
            state = AIState(
                entropy=1.0,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=100,
                max_context_length=2048,
                fatigue=0.0
            )
            gov.govern(state)

        fatigue_no_failure = gov.fatigue_accumulator.get_fatigue()

        # Reset and run with tool failures
        gov.reset()
        for i in range(5):
            state = AIState(
                entropy=1.0,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.ERROR,
                context_length=100,
                max_context_length=2048,
                fatigue=0.0
            )
            gov.govern(state)

        fatigue_with_failure = gov.fatigue_accumulator.get_fatigue()

        # Fatigue with failures should be higher
        assert fatigue_with_failure > fatigue_no_failure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
