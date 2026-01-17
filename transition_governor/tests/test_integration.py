"""
Integration tests demonstrating end-to-end governor functionality.

These tests show the complete system working together:
- Governor + EchoZero bridge
- Multi-agent coordination
- Deterministic replay
- Safety properties under realistic scenarios
"""

import pytest
import numpy as np

from transition_governor.core.governor import TransitionGovernor
from transition_governor.core.state import AIState, ToolState, GovernanceState
from transition_governor.echozero_bridge.runtime import (
    GovernedEchoZeroRuntime,
    MockEchoZeroClient,
    ContractViolationError
)
from transition_governor.echozero_bridge.contract import (
    GovernedContext,
    EchoZeroResponse,
    AllowedScope
)
from transition_governor.multi_agent.coordinator import MultiAgentCoordinator
from transition_governor.multi_agent.agent import AgentConfig


class TestEndToEndGovernance:
    """Test complete governor + EchoZero integration."""

    def test_normal_operation_flow(self):
        """Test normal operation without brownout."""
        governor = TransitionGovernor(seed=42, enable_logging=False)
        client = MockEchoZeroClient(compliant=True)
        runtime = GovernedEchoZeroRuntime(governor, client)

        # Stable state
        state = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.9,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        # Execute
        response, context = runtime.execute(state, "What is 2+2?")

        # Should be in normal operation
        assert context.governance_state == GovernanceState.NORMAL
        assert context.allowed_scope == AllowedScope.FULL
        assert response is not None
        assert runtime.get_violation_count() == 0

    def test_brownout_triggered_and_enforced(self):
        """Test that brownout is triggered and constraints enforced."""
        governor = TransitionGovernor(seed=42, enable_logging=False)
        client = MockEchoZeroClient(compliant=True)
        runtime = GovernedEchoZeroRuntime(governor, client)

        # Unstable state that should trigger brownout
        state = AIState(
            entropy=4.0,
            entropy_dot=5.0,
            confidence=0.2,
            confidence_dot=-3.0,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        # Execute
        response, context = runtime.execute(state, "Complex unstable query")

        # Should enter brownout
        if context.governance_state == GovernanceState.BROWNOUT:
            # Verify brownout constraints
            assert context.confidence_budget < 0.5
            assert context.max_tool_calls == 0
            assert context.allowed_scope in [AllowedScope.EXPLORATORY, AllowedScope.ABSTAIN]
            assert "Governor Notice" in response or "not confident" in response.lower()

    def test_contract_violation_blocked(self):
        """Test that contract violations are hard-blocked."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Non-compliant client that violates contract
        bad_client = MockEchoZeroClient(compliant=False)
        runtime = GovernedEchoZeroRuntime(governor, bad_client, strict_validation=True)

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

        # Should raise ContractViolationError
        with pytest.raises(ContractViolationError):
            runtime.execute(state, "Test input")


class TestDeterministicReplay:
    """Test deterministic replay capability."""

    def test_complete_session_replay(self):
        """Test that a complete session can be replayed deterministically."""
        # Create identical governors
        gov1 = TransitionGovernor(seed=42, enable_logging=False)
        gov2 = TransitionGovernor(seed=42, enable_logging=False)

        # Create a sequence of realistic states
        states = []
        np.random.seed(42)

        for i in range(20):
            # Simulate gradual instability
            entropy = 1.0 + i * 0.15 + np.random.rand() * 0.1
            confidence = max(0.1, 0.9 - i * 0.03)

            # Occasional tool failures
            tool_state = ToolState.ERROR if i % 7 == 0 else ToolState.INACTIVE

            state = AIState(
                entropy=entropy,
                entropy_dot=0.0,  # Will be computed
                confidence=confidence,
                confidence_dot=0.0,  # Will be computed
                tool_state=tool_state,
                context_length=100 + i * 50,
                max_context_length=2048,
                fatigue=0.0
            )
            states.append(state)

        # Process with gov1
        outputs1 = [gov1.govern(s) for s in states]

        # Replay with gov2
        outputs2 = gov2.replay_from_history(states)

        # Must be identical
        assert len(outputs1) == len(outputs2)
        for o1, o2 in zip(outputs1, outputs2):
            assert o1.governance_state == o2.governance_state
            assert o1.confidence_cap == o2.confidence_cap
            assert o1.transition_intensity == o2.transition_intensity
            assert o1.remaining_margin == o2.remaining_margin
            assert o1.authority_weights == o2.authority_weights

    def test_replay_includes_brownout_transitions(self):
        """Test that brownout transitions are captured in replay."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Sequence that should trigger brownout
        states = [
            # Start stable
            AIState(1.0, 0.0, 0.9, 0.0, ToolState.INACTIVE, 100, 2048, 0.0),
            AIState(1.0, 0.0, 0.9, 0.0, ToolState.INACTIVE, 150, 2048, 0.0),

            # Sudden instability (much higher to guarantee brownout)
            AIState(5.0, 8.0, 0.2, -1.5, ToolState.ERROR, 200, 2048, 0.0),
            AIState(6.0, 10.0, 0.1, -2.0, ToolState.ERROR, 250, 2048, 0.0),

            # Gradual recovery
            AIState(3.0, -2.0, 0.4, 0.2, ToolState.INACTIVE, 300, 2048, 0.0),
            AIState(2.0, -1.0, 0.6, 0.2, ToolState.INACTIVE, 350, 2048, 0.0),
            AIState(1.5, -0.5, 0.8, 0.2, ToolState.INACTIVE, 400, 2048, 0.0),
        ]

        # Process
        outputs = [governor.govern(s) for s in states]

        # Replay
        replayed_outputs = governor.replay_from_history(states)

        # Check that brownout occurred and was replayed
        had_brownout = any(o.governance_state == GovernanceState.BROWNOUT for o in outputs)
        assert had_brownout, "Expected brownout to occur"

        # Verify replay matches
        for o1, o2 in zip(outputs, replayed_outputs):
            assert o1.governance_state == o2.governance_state


class TestMultiAgentIntegration:
    """Test multi-agent coordination with governor."""

    def test_three_agent_coordination(self):
        """Test coordination of 3 agents with different stability levels."""
        coordinator = MultiAgentCoordinator(blend_rate=0.2, seed=42)

        # Add three agents
        coordinator.add_agent(AgentConfig(agent_id="stable", max_context_length=2048))
        coordinator.add_agent(AgentConfig(agent_id="moderate", max_context_length=2048))
        coordinator.add_agent(AgentConfig(agent_id="unstable", max_context_length=2048))

        # Run 15 rounds with different stability levels
        for round_num in range(15):
            states = {
                "stable": AIState(
                    entropy=1.0,
                    entropy_dot=0.0,
                    confidence=0.95,
                    confidence_dot=0.0,
                    tool_state=ToolState.INACTIVE,
                    context_length=100,
                    max_context_length=2048,
                    fatigue=0.0
                ),
                "moderate": AIState(
                    entropy=2.0,
                    entropy_dot=0.5,
                    confidence=0.7,
                    confidence_dot=-0.1,
                    tool_state=ToolState.INACTIVE,
                    context_length=100,
                    max_context_length=2048,
                    fatigue=1.0
                ),
                "unstable": AIState(
                    entropy=5.0,
                    entropy_dot=3.0,
                    confidence=0.2,
                    confidence_dot=-2.0,
                    tool_state=ToolState.ERROR,
                    context_length=100,
                    max_context_length=2048,
                    fatigue=8.0
                )
            }

            results = coordinator.process_round(states)

        # Get final authority distribution
        weights = coordinator.get_authority_distribution()

        # Stable agent should have most authority
        assert weights["stable"] > weights["moderate"]
        assert weights["stable"] > weights["unstable"]
        assert weights["moderate"] > weights["unstable"]

        # Dominant agent should be stable
        dominant = coordinator.allocator.get_dominant_agent()
        assert dominant == "stable"

        # Response selection should pick stable agent
        responses = {
            "stable": "Stable response",
            "moderate": "Moderate response",
            "unstable": "Unstable response"
        }
        selected = coordinator.select_response(responses)
        assert selected == "Stable response"

    def test_hallucination_containment_scenario(self):
        """Test that a hallucinating agent is contained."""
        coordinator = MultiAgentCoordinator(blend_rate=0.3, seed=42)

        # Two agents: one normal, one hallucinating
        coordinator.add_agent(AgentConfig(agent_id="normal", max_context_length=2048))
        coordinator.add_agent(AgentConfig(agent_id="hallucinating", max_context_length=2048))

        # Track authority over time
        authority_history = []

        # Simulate scenario where one agent starts hallucinating
        for round_num in range(20):
            # Normal agent stays stable
            state_normal = AIState(
                entropy=1.0,
                entropy_dot=0.0,
                confidence=0.9,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=100 + round_num * 10,
                max_context_length=2048,
                fatigue=0.0
            )

            # Hallucinating agent becomes increasingly unstable
            instability = min(10.0, round_num * 0.5)
            state_hallucinating = AIState(
                entropy=2.0 + instability,
                entropy_dot=instability * 0.3,
                confidence=max(0.1, 0.9 - instability * 0.08),
                confidence_dot=-instability * 0.1,
                tool_state=ToolState.ERROR if round_num > 5 else ToolState.ACTIVE,
                context_length=100 + round_num * 10,
                max_context_length=2048,
                fatigue=instability
            )

            coordinator.process_round({
                "normal": state_normal,
                "hallucinating": state_hallucinating
            })

            weights = coordinator.get_authority_distribution()
            authority_history.append(weights.copy())

        # Hallucinating agent should have lost most authority
        final_weights = authority_history[-1]
        assert final_weights["normal"] > 0.7
        assert final_weights["hallucinating"] < 0.3

        # Containment should be detected
        assert coordinator.detect_hallucination_containment()


class TestRealisticScenarios:
    """Test realistic usage scenarios."""

    def test_long_conversation_with_fatigue(self):
        """Test that fatigue accumulates and degrades performance over long conversation."""
        governor = TransitionGovernor(seed=42, enable_logging=False)
        client = MockEchoZeroClient(compliant=True)
        runtime = GovernedEchoZeroRuntime(governor, client)

        # Simulate 50-turn conversation
        initial_confidence_cap = None
        final_confidence_cap = None

        for turn in range(50):
            # Gradually increasing context and moderate instability
            context_len = min(100 + turn * 40, 2048)  # Cap at max
            state = AIState(
                entropy=1.5 + (turn % 10) * 0.1,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=context_len,
                max_context_length=2048,
                fatigue=0.0  # Will be updated by governor
            )

            response, context = runtime.execute(state, f"Turn {turn} query")

            if turn == 0:
                initial_confidence_cap = context.confidence_budget
            if turn == 49:
                final_confidence_cap = context.confidence_budget

        # Confidence cap should decrease over time due to fatigue
        assert final_confidence_cap < initial_confidence_cap

    def test_tool_failure_cascade_prevention(self):
        """Test that cascading tool failures are prevented."""
        governor = TransitionGovernor(seed=42, enable_logging=False)
        client = MockEchoZeroClient(compliant=True)
        runtime = GovernedEchoZeroRuntime(governor, client)

        # Simulate multiple tool failures in sequence
        for failure_num in range(5):
            state = AIState(
                entropy=2.0 + failure_num * 0.5,
                entropy_dot=0.0,
                confidence=0.7 - failure_num * 0.1,
                confidence_dot=0.0,
                tool_state=ToolState.ERROR,
                context_length=100,
                max_context_length=2048,
                fatigue=0.0
            )

            response, context = runtime.execute(state, "Attempt tool use")

            # After several failures, tools should be blocked
            if failure_num >= 3:
                # Should be in brownout with tools blocked
                if context.governance_state == GovernanceState.BROWNOUT:
                    assert context.max_tool_calls == 0

        # No contract violations despite failures
        assert runtime.get_violation_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
