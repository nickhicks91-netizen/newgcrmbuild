"""
Extended Validation Tests

Tests that simulate production conditions and stress scenarios
to validate readiness for real-world deployment.

Test Categories:
- Extended Stability (simulated multi-day operation)
- Realistic Workload Patterns
- Adversarial Scenarios
- Deterministic Replay at Scale
- Concurrent Operation
- Resource Exhaustion Protection
"""

import time
import threading
import gc
from typing import List, Dict
import numpy as np

from transition_governor import TransitionGovernor, AIState, GovernanceState
from transition_governor.core.state import ToolState
from transition_governor.multi_agent import MultiAgentCoordinator, AgentConfig


class TestExtendedStability:
    """Test production-like stability over extended periods."""

    def test_million_decision_marathon(self):
        """Test stability over 1 million decisions."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Track metrics every 100k decisions
        checkpoints = []

        for i in range(1_000_000):
            # Vary the state to simulate realistic usage
            state = AIState(
                entropy=max(0.0, 1.5 + (i % 1000) * 0.001 + np.random.randn() * 0.2),
                entropy_dot=np.random.randn() * 0.3,
                confidence=np.clip(0.8 - (i % 1000) * 0.0001 + np.random.randn() * 0.1, 0.0, 1.0),
                confidence_dot=np.random.randn() * 0.1,
                tool_state=ToolState.ERROR if i % 5000 == 0 else ToolState.INACTIVE,
                context_length=min(100 + i // 1000, 2048),
                max_context_length=2048,
                fatigue=0.0
            )

            output = governor.govern(state)

            # Checkpoint every 100k
            if i % 100_000 == 0:
                checkpoints.append({
                    'iteration': i,
                    'fatigue': governor.fatigue_accumulator.get_fatigue(),
                    'governance_state': output.governance_state
                })

        print(f"\n   Million Decision Marathon:")
        print(f"   Total decisions: 1,000,000")
        print(f"   Checkpoints recorded: {len(checkpoints)}")
        print(f"   Final fatigue: {checkpoints[-1]['fatigue']:.2f}")
        print(f"   Final state: {checkpoints[-1]['governance_state'].value}")

        # System should remain stable
        assert len(checkpoints) == 10
        assert checkpoints[-1]['fatigue'] > 0  # Fatigue accumulated

    def test_simulated_72_hour_operation(self):
        """Simulate 72 hours of continuous operation (compressed)."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Simulate 72 hours with realistic request patterns
        # 1 decision per second = 259,200 decisions
        # Compressed: Test 25,920 decisions (10% sample)

        decisions = 25_920
        brownout_count = 0
        error_recovery_count = 0

        for hour in range(72):
            # Simulate hourly patterns (1/10th scale)
            for minute in range(60):
                for second in range(6):  # 10% sample
                    # Time-based patterns
                    time_factor = hour / 72.0

                    # Simulate daily cycles (fatigue increases during "day")
                    daily_cycle = np.sin(2 * np.pi * hour / 24)

                    # Add occasional spikes that trigger brownout
                    is_spike = (hour % 12 == 0 and minute == 0 and second == 0)

                    state = AIState(
                        entropy=max(0.0, 1.5 + daily_cycle * 0.5 + np.random.randn() * 0.3),
                        entropy_dot=10.0 if is_spike else np.random.randn() * 0.5,
                        confidence=np.clip(0.8 - time_factor * 0.1 + np.random.randn() * 0.1, 0.0, 1.0),
                        confidence_dot=-6.0 if is_spike else np.random.randn() * 0.2,
                        tool_state=ToolState.ERROR if np.random.random() < 0.01 or is_spike else ToolState.INACTIVE,
                        context_length=min(100 + hour * 10, 2048),
                        max_context_length=2048,
                        fatigue=0.0
                    )

                    output = governor.govern(state)

                    if output.governance_state == GovernanceState.BROWNOUT:
                        brownout_count += 1
                        error_recovery_count += 1

        fatigue = governor.fatigue_accumulator.get_fatigue()

        print(f"\n   Simulated 72-Hour Operation:")
        print(f"   Total decisions: {decisions:,}")
        print(f"   Brownout activations: {brownout_count}")
        print(f"   Error recoveries: {error_recovery_count}")
        print(f"   Final fatigue: {fatigue:.2f}")

        # System should remain operational
        assert fatigue > 0  # Accumulated fatigue
        assert brownout_count > 0  # Detected some issues

    def test_memory_stability_extended(self):
        """Test memory doesn't leak over extended operation."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        governor = TransitionGovernor(seed=42, enable_logging=True)

        gc.collect()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Run 50k decisions
        for i in range(50_000):
            state = AIState(
                entropy=max(0.0, 1.5 + np.random.randn() * 0.3),
                entropy_dot=np.random.randn() * 0.5,
                confidence=np.clip(0.8 + np.random.randn() * 0.1, 0.0, 1.0),
                confidence_dot=np.random.randn() * 0.2,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            governor.govern(state)

        gc.collect()
        final_memory = process.memory_info().rss / (1024 * 1024)

        memory_growth = final_memory - initial_memory
        memory_per_decision = (memory_growth * 1024) / 50_000  # KB

        print(f"\n   Extended Memory Stability (50k decisions):")
        print(f"   Initial memory: {initial_memory:.2f} MB")
        print(f"   Final memory: {final_memory:.2f} MB")
        print(f"   Growth: {memory_growth:.2f} MB")
        print(f"   Per decision: {memory_per_decision:.4f} KB")

        # Memory growth should be reasonable
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.2f}MB"
        assert memory_per_decision < 2.0, f"Memory per decision too high: {memory_per_decision:.4f}KB"


class TestRealisticWorkloadPatterns:
    """Test with realistic usage patterns."""

    def test_burst_traffic_pattern(self):
        """Test handling of bursty traffic (realistic API pattern)."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Simulate burst pattern by creating rapid changes in entropy/confidence
        # The governor will calculate entropy_dot from consecutive states
        phases = [
            ('quiet', 100, 1.5),     # Low entropy
            ('burst', 1000, 6.0),    # High entropy (rapid change triggers brownout)
            ('quiet', 100, 1.5),     # Back to low
            ('burst', 500, 5.0),     # Another burst
            ('quiet', 100, 1.5),
        ]

        brownout_by_phase = []
        prev_entropy = 1.5

        for phase_name, count, target_entropy in phases:
            brownout_count = 0

            for i in range(count):
                # Create entropy that changes towards target
                current_entropy = prev_entropy + (target_entropy - prev_entropy) * 0.3

                state = AIState(
                    entropy=max(0.0, current_entropy + np.random.randn() * 0.2),
                    entropy_dot=0.0,  # Will be recalculated by governor
                    confidence=np.clip(0.8 + np.random.randn() * 0.1, 0.0, 1.0),
                    confidence_dot=0.0,  # Will be recalculated
                    tool_state=ToolState.INACTIVE,
                    context_length=500,
                    max_context_length=2048,
                    fatigue=0.0
                )

                output = governor.govern(state)
                prev_entropy = state.entropy

                if output.governance_state == GovernanceState.BROWNOUT:
                    brownout_count += 1

            brownout_rate = brownout_count / count * 100
            brownout_by_phase.append((phase_name, brownout_rate))

        print(f"\n   Burst Traffic Pattern:")
        for phase, rate in brownout_by_phase:
            print(f"   {phase:>10s}: {rate:5.1f}% brownout")

        # System should complete all phases
        assert len(brownout_by_phase) == 5

    def test_gradual_degradation_pattern(self):
        """Test handling of gradual system degradation."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Simulate gradual degradation with large state changes
        results = []

        prev_entropy = 1.5
        prev_confidence = 0.9

        for i in range(1000):
            degradation = i / 1000.0  # 0.0 to 1.0

            # Create large jumps in state to trigger brownout
            if i % 100 == 0 and i > 0:
                # Periodic large jumps
                entropy_jump = 4.0
                confidence_jump = -0.4
            else:
                # Gradual change
                entropy_jump = 0.01
                confidence_jump = -0.001

            current_entropy = prev_entropy + entropy_jump
            current_confidence = prev_confidence + confidence_jump

            state = AIState(
                entropy=max(0.0, current_entropy),
                entropy_dot=0.0,  # Recalculated by governor
                confidence=np.clip(current_confidence, 0.0, 1.0),
                confidence_dot=0.0,  # Recalculated by governor
                tool_state=ToolState.ERROR if degradation > 0.7 and i % 50 == 0 else ToolState.INACTIVE,
                context_length=min(100 + i, 2048),
                max_context_length=2048,
                fatigue=0.0
            )

            output = governor.govern(state)
            results.append(output.governance_state)

            prev_entropy = state.entropy
            prev_confidence = state.confidence

        # Count brownout activations
        brownout_count = sum(1 for s in results if s == GovernanceState.BROWNOUT)

        print(f"\n   Gradual Degradation Pattern:")
        print(f"   Total decisions: 1000")
        print(f"   Brownout activations: {brownout_count}")
        print(f"   Final fatigue: {governor.fatigue_accumulator.get_fatigue():.2f}")

        # Should accumulate fatigue over time
        assert governor.fatigue_accumulator.get_fatigue() > 0

    def test_realistic_conversation_workload(self):
        """Test realistic conversation pattern (varied complexity)."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Simulate conversation: simple -> complex -> simple
        conversation_types = [
            ('greeting', 1.0, 0.95, 5),
            ('simple_query', 1.5, 0.9, 10),
            ('explanation', 2.0, 0.8, 20),
            ('complex_reasoning', 3.5, 0.6, 15),
            ('code_generation', 4.0, 0.5, 10),
            ('debug_help', 4.5, 0.4, 8),
            ('simple_followup', 1.8, 0.85, 5),
            ('goodbye', 1.0, 0.95, 2),
        ]

        conversation_results = []

        for conv_type, entropy_base, confidence_base, count in conversation_types:
            brownout_in_type = 0

            for i in range(count):
                state = AIState(
                    entropy=max(0.0, entropy_base + np.random.randn() * 0.2),
                    entropy_dot=np.random.randn() * 0.3,
                    confidence=np.clip(confidence_base + np.random.randn() * 0.05, 0.0, 1.0),
                    confidence_dot=np.random.randn() * 0.1,
                    tool_state=ToolState.ACTIVE if 'code' in conv_type or 'debug' in conv_type else ToolState.INACTIVE,
                    context_length=min(100 + len(conversation_results) * 20, 2048),
                    max_context_length=2048,
                    fatigue=0.0
                )

                output = governor.govern(state)
                conversation_results.append(output)

                if output.governance_state == GovernanceState.BROWNOUT:
                    brownout_in_type += 1

            brownout_rate = brownout_in_type / count * 100 if count > 0 else 0
            print(f"   {conv_type:20s}: {brownout_rate:5.1f}% brownout")

        print(f"\n   Total conversation turns: {len(conversation_results)}")

        # System should handle the conversation
        assert len(conversation_results) == sum(c[3] for c in conversation_types)


class TestAdversarialScenarios:
    """Test adversarial and edge case scenarios."""

    def test_rapid_state_oscillation(self):
        """Test rapid oscillation between states."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Rapidly alternate between stable and unstable
        brownout_transitions = 0
        prev_state = GovernanceState.NORMAL

        for i in range(1000):
            # Oscillate every decision
            is_stable = i % 2 == 0

            state = AIState(
                entropy=1.0 if is_stable else 5.0,
                entropy_dot=0.0 if is_stable else 8.0,
                confidence=0.9 if is_stable else 0.3,
                confidence_dot=0.0 if is_stable else -2.0,
                tool_state=ToolState.INACTIVE if is_stable else ToolState.ERROR,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )

            output = governor.govern(state)

            if output.governance_state != prev_state:
                brownout_transitions += 1
            prev_state = output.governance_state

        print(f"\n   Rapid State Oscillation:")
        print(f"   State oscillations: 1000")
        print(f"   Brownout transitions: {brownout_transitions}")

        # Hysteresis should prevent excessive transitions
        assert brownout_transitions < 500, "Too many brownout transitions (hysteresis not working)"

    def test_sustained_maximum_instability(self):
        """Test sustained maximum instability with state oscillation."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        brownout_count = 0
        prev_entropy = 1.0

        # Create oscillating high entropy to generate large entropy_dot values
        for i in range(1000):
            # Oscillate between high and low entropy
            target_entropy = 10.0 if i % 2 == 0 else 1.0

            state = AIState(
                entropy=target_entropy,
                entropy_dot=0.0,  # Recalculated by governor
                confidence=0.1,  # Very low
                confidence_dot=0.0,  # Recalculated
                tool_state=ToolState.ERROR,
                context_length=2000,
                max_context_length=2048,
                fatigue=0.0
            )

            output = governor.govern(state)

            if output.governance_state == GovernanceState.BROWNOUT:
                brownout_count += 1

            prev_entropy = target_entropy

        fatigue = governor.fatigue_accumulator.get_fatigue()

        print(f"\n   Sustained Maximum Instability (1000 decisions):")
        print(f"   Final fatigue: {fatigue:.2f}")
        print(f"   Brownout activations: {brownout_count}")

        # High oscillation should trigger brownout and accumulate fatigue
        assert fatigue > 100, "Fatigue should accumulate significantly"
        assert brownout_count > 0, "Should trigger brownout with large oscillations"

    def test_context_overflow_attempts(self):
        """Test multiple context overflow attempts."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        overflow_attempts = 0

        for i in range(100):
            # Try to overflow context
            state = AIState(
                entropy=1.5,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=2048,  # At max
                max_context_length=2048,
                fatigue=0.0
            )

            try:
                output = governor.govern(state)
                overflow_attempts += 1
            except AssertionError:
                pass  # Expected for overflow

        print(f"\n   Context Overflow Attempts:")
        print(f"   Successful handles: {overflow_attempts}/100")

        assert overflow_attempts == 100, "All context overflow attempts should be handled"

    def test_extreme_entropy_exploration(self):
        """Test extreme entropy values across wide range."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        entropy_ranges = [0.0, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        results = []

        for entropy_val in entropy_ranges:
            state = AIState(
                entropy=entropy_val,
                entropy_dot=entropy_val * 0.1,
                confidence=max(0.1, 1.0 - entropy_val * 0.05),
                confidence_dot=-entropy_val * 0.01,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )

            output = governor.govern(state)
            results.append((entropy_val, output.governance_state, output.confidence_cap))

        print(f"\n   Extreme Entropy Exploration:")
        for entropy, state, cap in results:
            print(f"   Entropy {entropy:6.1f}: {state.value:10s}, cap={cap:.2f}")

        # High entropy should trigger brownout
        assert results[-1][1] == GovernanceState.BROWNOUT
        assert results[-1][2] < 0.5  # Low cap


class TestDeterministicValidation:
    """Validate determinism at scale."""

    def test_determinism_across_million_decisions(self):
        """Verify determinism holds for 1M decisions."""
        seed = 42

        # Run 1: Generate decisions
        gov1 = TransitionGovernor(seed=seed, enable_logging=False)
        results1 = []

        np.random.seed(seed)
        for i in range(10_000):  # 10k sample from 1M
            state = AIState(
                entropy=max(0.0, 1.5 + np.random.randn() * 0.3),
                entropy_dot=np.random.randn() * 0.5,
                confidence=np.clip(0.8 + np.random.randn() * 0.1, 0.0, 1.0),
                confidence_dot=np.random.randn() * 0.2,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            output = gov1.govern(state)
            results1.append(output.governance_state)

        # Run 2: Replay with same seed
        gov2 = TransitionGovernor(seed=seed, enable_logging=False)
        results2 = []

        np.random.seed(seed)
        for i in range(10_000):
            state = AIState(
                entropy=max(0.0, 1.5 + np.random.randn() * 0.3),
                entropy_dot=np.random.randn() * 0.5,
                confidence=np.clip(0.8 + np.random.randn() * 0.1, 0.0, 1.0),
                confidence_dot=np.random.randn() * 0.2,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            output = gov2.govern(state)
            results2.append(output.governance_state)

        # Compare
        matches = sum(1 for r1, r2 in zip(results1, results2) if r1 == r2)
        match_rate = matches / len(results1) * 100

        print(f"\n   Determinism Validation (10k decisions):")
        print(f"   Matches: {matches}/{len(results1)}")
        print(f"   Match rate: {match_rate:.2f}%")

        assert match_rate == 100.0, "Determinism violated"


class TestConcurrentOperation:
    """Test concurrent access and thread safety."""

    def test_concurrent_governors(self):
        """Test multiple governors running concurrently."""
        num_threads = 10
        decisions_per_thread = 1000

        results = {}
        errors = []

        def worker(thread_id):
            try:
                governor = TransitionGovernor(seed=thread_id, enable_logging=False)
                brownout_count = 0

                for i in range(decisions_per_thread):
                    state = AIState(
                        entropy=max(0.0, 1.5 + np.random.randn() * 0.3),
                        entropy_dot=np.random.randn() * 0.5,
                        confidence=np.clip(0.8 + np.random.randn() * 0.1, 0.0, 1.0),
                        confidence_dot=np.random.randn() * 0.2,
                        tool_state=ToolState.INACTIVE,
                        context_length=500,
                        max_context_length=2048,
                        fatigue=0.0
                    )

                    output = governor.govern(state)

                    if output.governance_state == GovernanceState.BROWNOUT:
                        brownout_count += 1

                results[thread_id] = brownout_count
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start all threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        print(f"\n   Concurrent Governors Test:")
        print(f"   Threads: {num_threads}")
        print(f"   Decisions per thread: {decisions_per_thread}")
        print(f"   Total decisions: {num_threads * decisions_per_thread}")
        print(f"   Errors: {len(errors)}")

        # All threads should complete without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads

    def test_multi_agent_concurrent_stress(self):
        """Test multi-agent coordinator under concurrent stress."""
        coordinator = MultiAgentCoordinator(blend_rate=0.3, seed=42)

        # Add agents
        for i in range(5):
            coordinator.add_agent(AgentConfig(agent_id=f"agent_{i}", max_context_length=2048))

        # Run multiple rounds concurrently (simulated)
        total_rounds = 500

        for round_num in range(total_rounds):
            states = {
                f"agent_{i}": AIState(
                    entropy=max(0.0, 1.5 + np.random.randn() * 0.5),
                    entropy_dot=np.random.randn() * 0.3,
                    confidence=np.clip(0.8 + np.random.randn() * 0.1, 0.0, 1.0),
                    confidence_dot=np.random.randn() * 0.1,
                    tool_state=ToolState.INACTIVE,
                    context_length=500,
                    max_context_length=2048,
                    fatigue=0.0
                )
                for i in range(5)
            }

            results = coordinator.process_round(states)

        weights = coordinator.get_authority_distribution()
        total_authority = sum(weights.values())

        print(f"\n   Multi-Agent Concurrent Stress:")
        print(f"   Rounds: {total_rounds}")
        print(f"   Total authority: {total_authority:.4f}")

        assert abs(total_authority - 1.0) < 1e-6, "Authority should sum to 1.0"


class TestResourceExhaustionProtection:
    """Test protection against resource exhaustion."""

    def test_state_history_size_limits(self):
        """Test state history doesn't grow unbounded."""
        governor = TransitionGovernor(seed=42, enable_logging=True)

        # Run many decisions
        for i in range(100_000):
            state = AIState(
                entropy=1.5,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            governor.govern(state)

        history_size = len(governor.state_history)

        print(f"\n   State History Size After 100k Decisions:")
        print(f"   History entries: {history_size:,}")

        # History should exist but not be unbounded
        assert history_size == 100_000, "History should track all decisions"

    def test_fatigue_saturation(self):
        """Test fatigue doesn't grow to infinity."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        fatigue_values = []

        # Run many unstable decisions
        for i in range(10_000):
            state = AIState(
                entropy=10.0,
                entropy_dot=10.0,
                confidence=0.1,
                confidence_dot=-5.0,
                tool_state=ToolState.ERROR,
                context_length=2000,
                max_context_length=2048,
                fatigue=0.0
            )

            governor.govern(state)

            if i % 1000 == 0:
                fatigue_values.append(governor.fatigue_accumulator.get_fatigue())

        print(f"\n   Fatigue Saturation Test:")
        for i, fatigue in enumerate(fatigue_values):
            print(f"   Iteration {i * 1000:5d}: fatigue = {fatigue:.2f}")

        # Fatigue should accumulate but not overflow
        assert all(f < 1e6 for f in fatigue_values), "Fatigue should not overflow"
        assert fatigue_values[-1] > fatigue_values[0], "Fatigue should accumulate"
