"""
Multi-agent coordination example.

This demonstrates:
- Multiple agents with local governors
- Global authority allocation
- Hallucination containment
- Authority-based response selection
"""

from transition_governor.multi_agent import MultiAgentCoordinator, AgentConfig
from transition_governor.core.state import AIState, ToolState


def main():
    print("=" * 70)
    print("MULTI-AGENT COORDINATION EXAMPLE")
    print("=" * 70)

    # Initialize coordinator
    print("\n1. Creating coordinator with 3 agents")
    coordinator = MultiAgentCoordinator(blend_rate=0.3, seed=42)

    # Add agents
    agent_configs = [
        ("stable_agent", "Consistently reliable agent"),
        ("moderate_agent", "Sometimes unstable"),
        ("unstable_agent", "Frequently hallucinating")
    ]

    for agent_id, description in agent_configs:
        coordinator.add_agent(AgentConfig(agent_id=agent_id, max_context_length=2048))
        print(f"   Added: {agent_id} - {description}")

    print("\n2. Running 15 coordination rounds...")
    print("   " + "-" * 65)

    for round_num in range(15):
        # Define states for each agent
        states = {
            "stable_agent": AIState(
                entropy=1.0,
                entropy_dot=0.0,
                confidence=0.95,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=100,
                max_context_length=2048,
                fatigue=0.0
            ),
            "moderate_agent": AIState(
                entropy=2.0,
                entropy_dot=0.5,
                confidence=0.7,
                confidence_dot=-0.1,
                tool_state=ToolState.INACTIVE,
                context_length=100,
                max_context_length=2048,
                fatigue=1.0
            ),
            "unstable_agent": AIState(
                entropy=5.0 + round_num * 0.2,  # Getting worse
                entropy_dot=3.0,
                confidence=max(0.1, 0.5 - round_num * 0.03),
                confidence_dot=-2.0,
                tool_state=ToolState.ERROR,
                context_length=100,
                max_context_length=2048,
                fatigue=5.0 + round_num * 0.5
            )
        }

        # Process round
        results = coordinator.process_round(states)

        # Print status every 5 rounds
        if round_num % 5 == 4:
            weights = results["authority_weights"]
            print(f"   Round {round_num + 1}:")
            print(f"     Stable:   authority={weights['stable_agent']:.3f}")
            print(f"     Moderate: authority={weights['moderate_agent']:.3f}")
            print(f"     Unstable: authority={weights['unstable_agent']:.3f}")

    print("   " + "-" * 65)

    print("\n3. Final authority distribution:")
    final_weights = coordinator.get_authority_distribution()
    for agent_id in ["stable_agent", "moderate_agent", "unstable_agent"]:
        authority = final_weights[agent_id]
        bar = "█" * int(authority * 50)
        print(f"   {agent_id:15s} [{authority:.3f}] {bar}")

    print("\n4. Response selection (authority-based, no voting):")
    responses = {
        "stable_agent": "Based on analysis, the answer is X.",
        "moderate_agent": "I think the answer might be Y.",
        "unstable_agent": "I'm absolutely certain it's Z with 100% confidence!"
    }

    selected = coordinator.select_response(responses)
    print(f"   Selected response (from highest authority agent):")
    print(f"   → {selected}")

    print("\n5. Hallucination containment verification:")
    is_contained = coordinator.detect_hallucination_containment()
    print(f"   Containment active: {is_contained}")

    if is_contained:
        print("   ✅ Unstable agent successfully contained")
        print("   ✅ Authority redistributed to stable agents")
        print("   ✅ No voting or majority rule used")

    print("\n✅ Multi-agent example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
