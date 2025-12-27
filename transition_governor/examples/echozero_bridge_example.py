"""
EchoZero bridge example.

This demonstrates:
- Governed EchoZero runtime
- Contract enforcement
- Brownout notifications
- Contract violation handling
"""

from transition_governor.core.governor import TransitionGovernor
from transition_governor.core.state import AIState, ToolState, GovernanceState
from transition_governor.echozero_bridge.runtime import (
    GovernedEchoZeroRuntime,
    MockEchoZeroClient,
    ContractViolationError
)
from transition_governor.echozero_bridge.contract import AllowedScope


def main():
    print("=" * 70)
    print("ECHOZERO BRIDGE EXAMPLE")
    print("=" * 70)

    # Initialize components
    print("\n1. Setting up governed runtime")
    governor = TransitionGovernor(seed=42)
    compliant_client = MockEchoZeroClient(compliant=True)
    runtime = GovernedEchoZeroRuntime(governor, compliant_client)
    print("   ✓ Governor initialized")
    print("   ✓ EchoZero client wrapped")

    # Test 1: Normal operation
    print("\n2. Testing normal operation (stable state)")
    state_normal = AIState(
        entropy=1.0,
        entropy_dot=0.0,
        confidence=0.9,
        confidence_dot=0.0,
        tool_state=ToolState.INACTIVE,
        context_length=100,
        max_context_length=2048,
        fatigue=0.0
    )

    response, context = runtime.execute(state_normal, "What is the capital of France?")
    print(f"   Governance State: {context.governance_state.value}")
    print(f"   Allowed Scope: {context.allowed_scope.value}")
    print(f"   Confidence Budget: {context.confidence_budget:.2f}")
    print(f"   Max Tool Calls: {context.max_tool_calls}")
    print(f"   Response: {response[:80]}...")

    # Test 2: Brownout triggered
    print("\n3. Testing brownout mode (unstable state)")
    state_unstable = AIState(
        entropy=5.0,
        entropy_dot=8.0,
        confidence=0.2,
        confidence_dot=-3.0,
        tool_state=ToolState.ERROR,
        context_length=100,
        max_context_length=2048,
        fatigue=0.0
    )

    response, context = runtime.execute(state_unstable, "Make a complex decision")
    print(f"   Governance State: {context.governance_state.value}")
    print(f"   Allowed Scope: {context.allowed_scope.value}")
    print(f"   Confidence Budget: {context.confidence_budget:.2f}")
    print(f"   Max Tool Calls: {context.max_tool_calls}")

    if context.governance_state == GovernanceState.BROWNOUT:
        print("\n   ⚠️  BROWNOUT MODE ACTIVE")
        print("   EchoZero received governed context with:")
        print(f"   - Confidence capped at {context.confidence_budget:.2f}")
        print(f"   - Tool calls: {context.max_tool_calls} (blocked)")
        print(f"   - Scope: {context.allowed_scope.value}")

        if "Governor Notice" in response:
            print("\n   User received brownout explanation:")
            notice_start = response.index("[Governor Notice:")
            notice_end = response.index("]", notice_start) + 1
            print(f"   {response[notice_start:notice_end]}")

    # Test 3: Contract violation
    print("\n4. Testing contract enforcement (non-compliant client)")
    bad_client = MockEchoZeroClient(compliant=False)
    strict_runtime = GovernedEchoZeroRuntime(
        governor=TransitionGovernor(seed=42),
        echozero_client=bad_client,
        strict_validation=True
    )

    state_test = AIState(
        entropy=1.0,
        entropy_dot=0.0,
        confidence=0.8,
        confidence_dot=0.0,
        tool_state=ToolState.INACTIVE,
        context_length=100,
        max_context_length=2048,
        fatigue=0.0
    )

    try:
        response, context = strict_runtime.execute(state_test, "Test query")
        print("   ❌ Contract violation was NOT caught (error!)")
    except ContractViolationError as e:
        print("   ✅ Contract violation detected and blocked!")
        print(f"   Violation: {str(e)}")

    # Test 4: Gradual degradation
    print("\n5. Testing gradual degradation over conversation")
    runtime_fresh = GovernedEchoZeroRuntime(
        governor=TransitionGovernor(seed=42),
        echozero_client=MockEchoZeroClient(compliant=True)
    )

    print("   Turn | Fatigue | Confidence Cap | State")
    print("   " + "-" * 50)

    for turn in range(0, 30, 5):
        state = AIState(
            entropy=1.5 + turn * 0.05,
            entropy_dot=0.1,
            confidence=0.8,
            confidence_dot=-0.05,
            tool_state=ToolState.INACTIVE,
            context_length=min(100 + turn * 60, 2048),
            max_context_length=2048,
            fatigue=0.0
        )

        _, context = runtime_fresh.execute(state, f"Turn {turn}")
        fatigue = runtime_fresh.governor.fatigue_accumulator.get_fatigue()

        print(f"   {turn:4d} | {fatigue:7.2f} | {context.confidence_budget:14.2f} | {context.governance_state.value}")

    print("\n6. Summary:")
    print("   ✓ Normal operation: full capabilities")
    print("   ✓ Brownout mode: reduced capabilities, user notified")
    print("   ✓ Contract violations: hard-blocked")
    print("   ✓ Gradual degradation: confidence decreases with fatigue")
    print("   ✓ EchoZero cannot override governor decisions")

    print("\n✅ EchoZero bridge example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
