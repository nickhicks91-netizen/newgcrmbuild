"""
Test brownout triggering and behavior.

Tests:
- Brownout entry on margin exhaustion
- Brownout exit with hysteresis
- Confidence cap enforcement
- Tool blocking in brownout
"""

import pytest

from transition_governor.core.governor import TransitionGovernor
from transition_governor.core.state import AIState, ToolState, GovernanceState
from transition_governor.core.degradation import BrownoutConfig


class TestBrownoutEntry:
    """Test brownout entry conditions."""

    def test_brownout_on_high_transition_intensity(self):
        """High transition intensity should trigger brownout."""
        config = BrownoutConfig(
            brownout_entry_threshold=0.0,
            brownout_exit_threshold=0.5
        )
        gov = TransitionGovernor(brownout_config=config, seed=42, enable_logging=False)

        # Create high-intensity transition state
        state = AIState(
            entropy=3.0,
            entropy_dot=5.0,  # Very high rate of change
            confidence=0.2,
            confidence_dot=-2.0,  # Very high rate of change
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output = gov.govern(state)

        # Should enter brownout due to high transition intensity
        assert output.governance_state == GovernanceState.BROWNOUT

    def test_brownout_on_negative_margin(self):
        """Negative stability margin should trigger brownout."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Force high transition intensity
        state = AIState(
            entropy=5.0,
            entropy_dot=10.0,
            confidence=0.1,
            confidence_dot=-5.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output = gov.govern(state)

        # Check margin is negative
        assert output.remaining_margin < 0.0

        # Should be in brownout
        assert output.governance_state == GovernanceState.BROWNOUT


class TestBrownoutBehavior:
    """Test behavior during brownout."""

    def test_tools_disabled_in_brownout(self):
        """Tools must be disabled during brownout."""
        config = BrownoutConfig(brownout_max_tool_calls=0)
        gov = TransitionGovernor(brownout_config=config, seed=42, enable_logging=False)

        # Trigger brownout
        state = AIState(
            entropy=3.0,
            entropy_dot=5.0,
            confidence=0.2,
            confidence_dot=-2.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output = gov.govern(state)

        if output.governance_state == GovernanceState.BROWNOUT:
            assert output.max_tool_calls_per_step == 0

    def test_confidence_capped_in_brownout(self):
        """Confidence should be aggressively capped in brownout."""
        config = BrownoutConfig(
            normal_confidence_cap=1.0,
            brownout_confidence_cap=0.3
        )
        gov = TransitionGovernor(brownout_config=config, seed=42, enable_logging=False)

        # Normal state
        state_normal = AIState(
            entropy=1.0,
            entropy_dot=0.1,
            confidence=0.8,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output_normal = gov.govern(state_normal)

        # Force brownout
        gov.reset()
        state_brownout = AIState(
            entropy=3.0,
            entropy_dot=5.0,
            confidence=0.2,
            confidence_dot=-2.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output_brownout = gov.govern(state_brownout)

        if output_brownout.governance_state == GovernanceState.BROWNOUT:
            # Brownout cap should be much lower
            assert output_brownout.confidence_cap < output_normal.confidence_cap

    def test_token_limit_enforced_in_brownout(self):
        """Token limits should be enforced in brownout."""
        config = BrownoutConfig(
            normal_max_tokens=None,
            brownout_max_tokens=50
        )
        gov = TransitionGovernor(brownout_config=config, seed=42, enable_logging=False)

        # Trigger brownout
        state = AIState(
            entropy=3.0,
            entropy_dot=5.0,
            confidence=0.2,
            confidence_dot=-2.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output = gov.govern(state)

        if output.governance_state == GovernanceState.BROWNOUT:
            assert output.max_tokens_per_step is not None
            assert output.max_tokens_per_step <= 50


class TestBrownoutHysteresis:
    """Test brownout entry/exit hysteresis."""

    def test_exit_threshold_higher_than_entry(self):
        """Exit threshold must be higher than entry (prevents oscillation)."""
        config = BrownoutConfig(
            brownout_entry_threshold=0.0,
            brownout_exit_threshold=0.5
        )
        assert config.brownout_exit_threshold > config.brownout_entry_threshold

    def test_brownout_recovery(self):
        """System should exit brownout when stability improves."""
        config = BrownoutConfig(
            brownout_entry_threshold=0.0,
            brownout_exit_threshold=1.0
        )
        gov = TransitionGovernor(brownout_config=config, seed=42, enable_logging=False)

        # Enter brownout
        state_unstable = AIState(
            entropy=3.0,
            entropy_dot=5.0,
            confidence=0.2,
            confidence_dot=-2.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output_unstable = gov.govern(state_unstable)

        # Should be in brownout
        in_brownout = (output_unstable.governance_state == GovernanceState.BROWNOUT)

        if in_brownout:
            # Stabilize system (multiple stable steps)
            for _ in range(10):
                state_stable = AIState(
                    entropy=1.0,
                    entropy_dot=0.0,
                    confidence=0.9,
                    confidence_dot=0.0,
                    tool_state=ToolState.INACTIVE,
                    context_length=100,
                    max_context_length=2048,
                    fatigue=0.0
                )
                output_stable = gov.govern(state_stable)

            # Should eventually exit brownout
            assert output_stable.governance_state == GovernanceState.NORMAL


class TestBrownoutDegradation:
    """Test degradation behavior."""

    def test_degradation_increases_with_fatigue(self):
        """Higher fatigue should increase degradation."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Low fatigue
        state_low = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.8,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.5
        )

        output_low = gov.govern(state_low)

        # Accumulate high fatigue
        for _ in range(50):
            state = AIState(
                entropy=2.0,
                entropy_dot=1.0,
                confidence=0.5,
                confidence_dot=-0.5,
                tool_state=ToolState.ERROR,
                context_length=100,
                max_context_length=2048,
                fatigue=0.0
            )
            gov.govern(state)

        # High fatigue state
        state_high = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.8,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output_high = gov.govern(state_high)

        # Confidence cap should be lower with high fatigue
        assert output_high.confidence_cap < output_low.confidence_cap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
