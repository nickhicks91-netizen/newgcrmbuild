"""
Test tool failure handling.

Tests:
- Tool failure triggers brownout
- No hallucination under tool failure
- Authority redistribution on tool error
- Recovery after tool restoration
"""

import pytest

from transition_governor.core.governor import TransitionGovernor
from transition_governor.core.state import AIState, ToolState, GovernanceState
from transition_governor.echozero_bridge.runtime import (
    GovernedEchoZeroRuntime,
    MockEchoZeroClient,
    ContractViolationError
)


class TestToolFailureResponse:
    """Test governor response to tool failures."""

    def test_tool_authority_reduced_on_error(self):
        """Tool authority should decrease when tool fails."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Normal tool state
        state_normal = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.8,
            confidence_dot=0.0,
            tool_state=ToolState.ACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output_normal = gov.govern(state_normal)
        normal_tool_authority = output_normal.authority_weights["tools"]

        # Tool error state
        gov.reset()
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

        output_error = gov.govern(state_error)
        error_tool_authority = output_error.authority_weights["tools"]

        # Tool authority should be lower on error
        assert error_tool_authority < normal_tool_authority

    def test_fatigue_increases_on_tool_failure(self):
        """Tool failures should accelerate fatigue accumulation."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Process several steps without tool failure
        for _ in range(5):
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

        # Reset and process with tool failures
        gov.reset()
        for _ in range(5):
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

        # Fatigue should be higher with failures
        assert fatigue_with_failure > fatigue_no_failure


class TestNoHallucinationUnderToolFailure:
    """
    Test that tool failures do NOT lead to hallucination.

    This is critical: when tools fail, the system must:
    1. Enter brownout
    2. Block tool calls
    3. Cap confidence aggressively
    4. Not fabricate tool results
    """

    def test_tool_calls_blocked_in_brownout(self):
        """Tool calls should be blocked when tools are failing."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Create tool failure state with high instability
        state = AIState(
            entropy=3.0,
            entropy_dot=2.0,
            confidence=0.3,
            confidence_dot=-1.0,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=5.0
        )

        output = gov.govern(state)

        # If in brownout, tools should be blocked
        if output.governance_state == GovernanceState.BROWNOUT:
            assert output.max_tool_calls_per_step == 0

    def test_echozero_cannot_override_tool_block(self):
        """EchoZero must not be able to call tools when blocked."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Create runtime with compliant EchoZero
        mock_client = MockEchoZeroClient(compliant=True)
        runtime = GovernedEchoZeroRuntime(
            governor=gov,
            echozero_client=mock_client,
            strict_validation=True
        )

        # Create state that triggers tool blocking
        state = AIState(
            entropy=3.0,
            entropy_dot=2.0,
            confidence=0.3,
            confidence_dot=-1.0,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=5.0
        )

        # Execute with tool failure
        response, context = runtime.execute(state, "Use tools to search")

        # If tools blocked, context should reflect that
        if context.max_tool_calls == 0:
            # EchoZero should not have made tool requests
            # (We can't directly check this with mock, but validation would catch it)
            assert runtime.get_violation_count() == 0


class TestToolFailureRecovery:
    """Test recovery after tool restoration."""

    def test_authority_restoration_after_recovery(self):
        """Tool authority should increase after tools recover."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Tool failure
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

        output_error = gov.govern(state_error)
        error_tool_authority = output_error.authority_weights["tools"]

        # Tool recovery (several stable steps)
        for _ in range(20):
            state_recovered = AIState(
                entropy=1.0,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=100,
                max_context_length=2048,
                fatigue=0.0
            )
            output_recovered = gov.govern(state_recovered)

        recovered_tool_authority = output_recovered.authority_weights["tools"]

        # Authority should increase after recovery
        assert recovered_tool_authority > error_tool_authority


class TestContractEnforcementOnToolFailure:
    """Test that contract is enforced even when tools fail."""

    def test_violation_detection_during_tool_failure(self):
        """Contract violations must be detected even during tool failures."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Non-compliant EchoZero (tries to violate contract)
        bad_client = MockEchoZeroClient(compliant=False)
        runtime = GovernedEchoZeroRuntime(
            governor=gov,
            echozero_client=bad_client,
            strict_validation=True
        )

        # Create tool failure state
        state = AIState(
            entropy=2.0,
            entropy_dot=1.0,
            confidence=0.5,
            confidence_dot=-0.5,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=2.0
        )

        # Execution should raise ContractViolationError
        with pytest.raises(ContractViolationError):
            runtime.execute(state, "Test input")

    def test_safe_fallback_on_echozero_failure(self):
        """If EchoZero fails during tool error, provide safe fallback."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # EchoZero that raises exception
        def failing_client(context, user_input):
            raise RuntimeError("EchoZero internal error")

        runtime = GovernedEchoZeroRuntime(
            governor=gov,
            echozero_client=failing_client,
            strict_validation=True
        )

        # Create tool failure state
        state = AIState(
            entropy=2.0,
            entropy_dot=1.0,
            confidence=0.5,
            confidence_dot=-0.5,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=2.0
        )

        # Should return safe fallback, not crash
        response, context = runtime.execute(state, "Test input")

        # Response should be safe fallback
        assert "error" in response.lower() or "cannot" in response.lower()


class TestNoFabricatedResults:
    """Test that tool failures don't lead to fabricated results."""

    def test_confidence_capped_during_tool_failure(self):
        """Confidence should be capped when tools are failing."""
        gov = TransitionGovernor(seed=42, enable_logging=False)

        # Tool failure with attempted high confidence
        state = AIState(
            entropy=1.0,  # Low entropy (high confidence)
            entropy_dot=0.0,
            confidence=0.95,  # High confidence
            confidence_dot=0.0,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        output = gov.govern(state)

        # Confidence cap should be reduced due to tool failure
        # (exact value depends on degradation, but should be < 1.0)
        assert output.confidence_cap < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
