"""
Test multi-agent coordination and authority allocation.

Tests:
- Authority redistribution based on stability
- Hallucination containment across agents
- No voting or consensus mechanisms
- Unstable agents lose influence smoothly
"""

import pytest
import numpy as np

from transition_governor.multi_agent.agent import GovernedAgent, AgentConfig
from transition_governor.multi_agent.allocator import GlobalAuthorityAllocator
from transition_governor.multi_agent.coordinator import MultiAgentCoordinator
from transition_governor.core.state import AIState, ToolState


class TestAuthorityAllocation:
    """Test global authority allocation."""

    def test_stable_agent_gets_more_authority(self):
        """More stable agents should receive higher authority."""
        allocator = GlobalAuthorityAllocator(seed=42)

        # Create agents
        agent1 = GovernedAgent(
            AgentConfig(agent_id="stable", max_context_length=2048),
            seed=42
        )
        agent2 = GovernedAgent(
            AgentConfig(agent_id="unstable", max_context_length=2048),
            seed=43
        )

        allocator.register_agent(agent1)
        allocator.register_agent(agent2)

        # Process stable state for agent1
        state1 = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.9,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )
        agent1.process_state(state1)

        # Process unstable state for agent2
        state2 = AIState(
            entropy=3.0,
            entropy_dot=5.0,
            confidence=0.2,
            confidence_dot=-2.0,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=5.0
        )
        agent2.process_state(state2)

        # Allocate authority
        weights = allocator.allocate_authority()

        # Stable agent should have more authority
        assert weights["stable"] > weights["unstable"]

    def test_authority_sums_to_one(self):
        """Total authority across agents must sum to 1.0."""
        allocator = GlobalAuthorityAllocator(seed=42)

        # Create multiple agents
        for i in range(5):
            agent = GovernedAgent(
                AgentConfig(agent_id=f"agent_{i}", max_context_length=2048),
                seed=42 + i
            )
            allocator.register_agent(agent)

            # Process random state
            state = AIState(
                entropy=np.random.rand() * 3.0,
                entropy_dot=np.random.randn() * 0.5,
                confidence=np.random.rand(),
                confidence_dot=np.random.randn() * 0.2,
                tool_state=ToolState.INACTIVE,
                context_length=100,
                max_context_length=2048,
                fatigue=np.random.rand() * 5.0
            )
            agent.process_state(state)

        # Allocate authority
        weights = allocator.allocate_authority()

        # Total should be 1.0
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6


class TestHallucinationContainment:
    """Test that hallucinating agents lose influence."""

    def test_unstable_agent_loses_authority(self):
        """Agent with high instability should lose authority over time."""
        coordinator = MultiAgentCoordinator(blend_rate=0.2, seed=42)

        # Add two agents
        agent1 = coordinator.add_agent(
            AgentConfig(agent_id="normal", max_context_length=2048)
        )
        agent2 = coordinator.add_agent(
            AgentConfig(agent_id="hallucinating", max_context_length=2048)
        )

        # Run several rounds
        for round_num in range(10):
            # Normal agent: stable states
            state1 = AIState(
                entropy=1.0,
                entropy_dot=0.0,
                confidence=0.9,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=100,
                max_context_length=2048,
                fatigue=0.0
            )

            # Hallucinating agent: very unstable
            state2 = AIState(
                entropy=5.0,
                entropy_dot=3.0,
                confidence=0.1,
                confidence_dot=-2.0,
                tool_state=ToolState.ERROR,
                context_length=100,
                max_context_length=2048,
                fatigue=10.0
            )

            results = coordinator.process_round({
                "normal": state1,
                "hallucinating": state2
            })

        # After several rounds, normal agent should dominate
        final_weights = coordinator.get_authority_distribution()
        assert final_weights["normal"] > final_weights["hallucinating"]

    def test_containment_detection(self):
        """Coordinator should detect successful containment."""
        coordinator = MultiAgentCoordinator(blend_rate=0.2, seed=42)

        # Add agents
        coordinator.add_agent(
            AgentConfig(agent_id="stable", max_context_length=2048)
        )
        coordinator.add_agent(
            AgentConfig(agent_id="unstable", max_context_length=2048)
        )

        # Run rounds with one unstable agent
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

            state_unstable = AIState(
                entropy=6.0,
                entropy_dot=4.0,
                confidence=0.1,
                confidence_dot=-3.0,
                tool_state=ToolState.ERROR,
                context_length=100,
                max_context_length=2048,
                fatigue=15.0
            )

            coordinator.process_round({
                "stable": state_stable,
                "unstable": state_unstable
            })

        # Containment should be working
        assert coordinator.detect_hallucination_containment()


class TestNoVotingOrConsensus:
    """Test that there is no voting or consensus mechanism."""

    def test_no_majority_rule(self):
        """Authority is based on stability, not majority."""
        coordinator = MultiAgentCoordinator(blend_rate=0.2, seed=42)

        # Add 3 unstable agents and 1 stable agent
        coordinator.add_agent(AgentConfig(agent_id="stable", max_context_length=2048))
        for i in range(3):
            coordinator.add_agent(
                AgentConfig(agent_id=f"unstable_{i}", max_context_length=2048)
            )

        # Run rounds
        for _ in range(10):
            # One stable state
            state_stable = AIState(
                entropy=1.0,
                entropy_dot=0.0,
                confidence=0.95,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=100,
                max_context_length=2048,
                fatigue=0.0
            )

            # Three unstable states
            state_unstable = AIState(
                entropy=4.0,
                entropy_dot=3.0,
                confidence=0.2,
                confidence_dot=-1.5,
                tool_state=ToolState.ERROR,
                context_length=100,
                max_context_length=2048,
                fatigue=8.0
            )

            states = {"stable": state_stable}
            for i in range(3):
                states[f"unstable_{i}"] = state_unstable

            coordinator.process_round(states)

        # The single stable agent should have high authority
        # despite being outnumbered 3:1
        weights = coordinator.get_authority_distribution()

        # Stable agent should have more authority than any single unstable agent
        for i in range(3):
            assert weights["stable"] > weights[f"unstable_{i}"]

    def test_response_selection_by_authority(self):
        """Response selection based on authority, not voting."""
        coordinator = MultiAgentCoordinator(blend_rate=0.3, seed=42)

        # Add agents
        coordinator.add_agent(AgentConfig(agent_id="A", max_context_length=2048))
        coordinator.add_agent(AgentConfig(agent_id="B", max_context_length=2048))

        # Process states (A is more stable)
        state_a = AIState(
            entropy=1.0,
            entropy_dot=0.0,
            confidence=0.95,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        state_b = AIState(
            entropy=3.0,
            entropy_dot=2.0,
            confidence=0.3,
            confidence_dot=-1.0,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=5.0
        )

        for _ in range(10):
            coordinator.process_round({"A": state_a, "B": state_b})

        # Select response
        responses = {
            "A": "Response from A",
            "B": "Response from B"
        }

        selected = coordinator.select_response(responses)

        # Should select response from agent A (more stable)
        assert selected == "Response from A"


class TestSmoothAuthorityTransition:
    """Test that authority transitions are smooth."""

    def test_authority_changes_gradually(self):
        """Authority should not jump instantly, but blend smoothly."""
        allocator = GlobalAuthorityAllocator(blend_rate=0.1, seed=42)

        # Create two agents
        agent1 = GovernedAgent(
            AgentConfig(agent_id="agent1", max_context_length=2048),
            seed=42
        )
        agent2 = GovernedAgent(
            AgentConfig(agent_id="agent2", max_context_length=2048),
            seed=43
        )

        allocator.register_agent(agent1)
        allocator.register_agent(agent2)

        # Initially equal states
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

        agent1.process_state(state)
        agent2.process_state(state)

        weights_initial = allocator.allocate_authority()

        # Now agent1 becomes very stable, agent2 very unstable
        state_stable = AIState(
            entropy=0.5,
            entropy_dot=0.0,
            confidence=0.99,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        state_unstable = AIState(
            entropy=5.0,
            entropy_dot=3.0,
            confidence=0.1,
            confidence_dot=-2.0,
            tool_state=ToolState.ERROR,
            context_length=100,
            max_context_length=2048,
            fatigue=10.0
        )

        # First iteration
        agent1.process_state(state_stable)
        agent2.process_state(state_unstable)
        weights_first = allocator.allocate_authority()

        # Change should be gradual, not instant
        change_first = abs(weights_first["agent1"] - weights_initial["agent1"])

        # Several more iterations
        for _ in range(10):
            agent1.process_state(state_stable)
            agent2.process_state(state_unstable)
            allocator.allocate_authority()

        weights_final = allocator.allocate_authority()
        change_final = abs(weights_final["agent1"] - weights_initial["agent1"])

        # Final change should be larger than first (accumulated)
        assert change_final > change_first

        # But first change should be relatively small (smooth)
        assert change_first < 0.5  # Not instant jump to 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
