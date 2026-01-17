"""
Basic example of Transition Governor usage.

This demonstrates:
- Creating a governor
- Processing states
- Observing brownout trigger
- Deterministic replay
"""

from transition_governor import TransitionGovernor, AIState, ToolState, GovernanceState


def main():
    print("=" * 70)
    print("TRANSITION GOVERNOR - BASIC EXAMPLE")
    print("=" * 70)

    # Initialize governor
    print("\n1. Initializing governor with seed=42 for determinism")
    governor = TransitionGovernor(seed=42)

    print("\n2. Processing stable state...")
    stable_state = AIState(
        entropy=1.0,
        entropy_dot=0.0,
        confidence=0.9,
        confidence_dot=0.0,
        tool_state=ToolState.INACTIVE,
        context_length=100,
        max_context_length=2048,
        fatigue=0.0
    )

    output = governor.govern(stable_state)
    print(f"   Governance State: {output.governance_state.value}")
    print(f"   Confidence Cap: {output.confidence_cap:.2f}")
    print(f"   Transition Intensity: {output.transition_intensity:.3f}")
    print(f"   Authority Weights: {output.authority_weights}")

    print("\n3. Processing unstable state (should trigger brownout)...")
    unstable_state = AIState(
        entropy=5.0,
        entropy_dot=8.0,
        confidence=0.2,
        confidence_dot=-2.0,
        tool_state=ToolState.ERROR,
        context_length=100,
        max_context_length=2048,
        fatigue=0.0
    )

    output = governor.govern(unstable_state)
    print(f"   Governance State: {output.governance_state.value}")
    print(f"   Confidence Cap: {output.confidence_cap:.2f}")
    print(f"   Transition Intensity: {output.transition_intensity:.3f}")
    print(f"   Remaining Margin: {output.remaining_margin:.3f}")
    print(f"   Max Tool Calls: {output.max_tool_calls_per_step}")
    print(f"   Authority Weights: {output.authority_weights}")

    if output.governance_state == GovernanceState.BROWNOUT:
        print("\n   ⚠️  BROWNOUT ACTIVE - System entered reduced-confidence mode")
        print("       - Confidence capped aggressively")
        print("       - Tools blocked")
        print("       - Output will be conservative")

    print("\n4. Demonstrating deterministic replay...")
    governor.reset()

    states = [stable_state, unstable_state]
    outputs_1 = [governor.govern(s) for s in states]

    outputs_2 = governor.replay_from_history(states)

    print(f"   Original outputs match replay: {outputs_1 == outputs_2}")
    print(f"   Governance states: {[o.governance_state.value for o in outputs_1]}")

    print("\n5. Fatigue accumulation over time...")
    governor.reset()

    for i in range(10):
        state = AIState(
            entropy=1.5,
            entropy_dot=0.2,
            confidence=0.8,
            confidence_dot=-0.1,
            tool_state=ToolState.INACTIVE,
            context_length=100 + i * 50,
            max_context_length=2048,
            fatigue=0.0
        )
        output = governor.govern(state)

        if i % 3 == 0:
            fatigue = governor.fatigue_accumulator.get_fatigue()
            print(f"   Step {i}: fatigue={fatigue:.2f}, confidence_cap={output.confidence_cap:.2f}")

    print("\n✅ Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
