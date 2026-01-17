"""
Identity Vector Test Suite

Tests the sparse identity vector for the Tachyon layer:
- Soft reinforcement updates
- Temporal decay dynamics
- Sparsity preservation
- Dominant pattern tracking
- Statistics computation
- Reset functionality
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/GRCM')

from grcm.echozero.tachyon.identity_vector import IdentityVector


def test_initialization():
    """Test identity vector initializes correctly."""
    print("\n" + "="*70)
    print("TEST 1: INITIALIZATION")
    print("="*70)

    identity = IdentityVector(num_patterns=10, decay_rate=0.01, update_strength=0.1)

    # Check initial state
    assert identity.state.shape == (10,), f"Wrong shape: {identity.state.shape}"
    assert np.allclose(identity.state, 0), "State should start at zero"
    assert identity.decay_rate == 0.01, "Wrong decay rate"
    assert identity.update_strength == 0.1, "Wrong update strength"

    print(f"✓ Initialized: {identity}")
    print(f"  State shape: {identity.state.shape}")
    print(f"  Initial values: all zeros")
    print(f"  Decay rate: {identity.decay_rate}")
    print(f"  Update strength: {identity.update_strength}")

    print("\n✅ TEST 1 PASSED")


def test_reinforcement():
    """Test soft reinforcement updates."""
    print("\n" + "="*70)
    print("TEST 2: SOFT REINFORCEMENT")
    print("="*70)

    identity = IdentityVector(num_patterns=5, update_strength=0.2)

    # Reinforce pattern 2
    identity.reinforce(2)

    assert identity.state[2] > 0, "Pattern not reinforced"
    assert identity.state[2] <= 1.0, "Value should be clipped to [0,1]"
    assert identity.total_updates == 1, "Update count not incremented"

    first_value = identity.state[2]
    print(f"✓ After 1 update: pattern[2] = {first_value:.4f}")

    # Reinforce again (should increase but saturate)
    identity.reinforce(2)
    second_value = identity.state[2]

    assert second_value > first_value, "Second update should increase value"
    assert second_value <= 1.0, "Value should be clipped"

    print(f"✓ After 2 updates: pattern[2] = {second_value:.4f}")

    # Reinforce many times (should saturate towards 1.0)
    for _ in range(10):
        identity.reinforce(2)

    saturated_value = identity.state[2]
    assert saturated_value > 0.9, f"Should saturate near 1.0, got {saturated_value:.4f}"

    print(f"✓ After 12 updates: pattern[2] = {saturated_value:.4f} (saturated)")

    print("\n✅ TEST 2 PASSED")


def test_temporal_decay():
    """Test temporal decay dynamics."""
    print("\n" + "="*70)
    print("TEST 3: TEMPORAL DECAY")
    print("="*70)

    identity = IdentityVector(num_patterns=5, decay_rate=0.1, update_strength=0.5)

    # Reinforce pattern 1 to high value
    for _ in range(5):
        identity.reinforce(1)

    initial_value = identity.state[1]
    print(f"✓ Initial value: {initial_value:.4f}")

    # Apply decay for 10 steps
    values = [initial_value]
    for _ in range(10):
        identity.decay()
        values.append(identity.state[1])

    final_value = identity.state[1]

    assert final_value < initial_value, "Decay should reduce value"
    assert final_value > 0, "Should not decay to exactly zero immediately"

    print(f"✓ After 10 decay steps: {final_value:.4f}")
    print(f"  Decay ratio: {final_value/initial_value:.2%}")

    # Check exponential decay shape
    # Value should decrease monotonically
    for i in range(len(values)-1):
        assert values[i] >= values[i+1], f"Non-monotonic decay at step {i}"

    print(f"✓ Decay is monotonic")

    print("\n✅ TEST 3 PASSED")


def test_sparsity():
    """Test that identity maintains sparsity."""
    print("\n" + "="*70)
    print("TEST 4: SPARSITY PRESERVATION")
    print("="*70)

    identity = IdentityVector(num_patterns=20, decay_rate=0.05, update_strength=0.1)

    # Reinforce only 3 patterns
    for _ in range(5):
        identity.reinforce(3)
        identity.reinforce(7)
        identity.reinforce(15)

    # Apply decay to push weak values to zero
    for _ in range(10):
        identity.decay()

    # Check sparsity
    active = np.sum(identity.state > 1e-6)
    sparsity = 1 - (active / len(identity.state))

    print(f"✓ Active dimensions: {active}/20")
    print(f"✓ Sparsity: {sparsity:.1%}")

    assert active <= 5, f"Too many active dimensions: {active}"
    assert sparsity >= 0.7, f"Sparsity too low: {sparsity:.2%}"

    # Get statistics
    stats = identity.get_statistics()
    print(f"✓ Statistics: {stats}")

    assert stats['sparsity'] == sparsity, "Statistics mismatch"
    assert stats['active_dimensions'] == active, "Active count mismatch"

    print("\n✅ TEST 4 PASSED")


def test_dominant_patterns():
    """Test dominant pattern extraction."""
    print("\n" + "="*70)
    print("TEST 5: DOMINANT PATTERN TRACKING")
    print("="*70)

    identity = IdentityVector(num_patterns=10)

    # Create known activation pattern
    identity.reinforce(2, strength=0.5)
    identity.reinforce(5, strength=0.3)
    identity.reinforce(7, strength=0.1)

    # Get dominant patterns
    dominant = identity.get_dominant_patterns(top_k=3)

    print(f"✓ Dominant patterns: {dominant}")

    # Check ordering (should be sorted by activation)
    assert len(dominant) == 3, f"Wrong number returned: {len(dominant)}"
    assert dominant[0][0] == 2, f"Top pattern should be 2, got {dominant[0][0]}"
    assert dominant[1][0] == 5, f"Second pattern should be 5, got {dominant[1][0]}"
    assert dominant[2][0] == 7, f"Third pattern should be 7, got {dominant[2][0]}"

    # Check values are sorted descending
    assert dominant[0][1] >= dominant[1][1], "Not sorted descending"
    assert dominant[1][1] >= dominant[2][1], "Not sorted descending"

    print(f"✓ Patterns correctly sorted by activation")

    # Test with top_k larger than active patterns
    all_dominant = identity.get_dominant_patterns(top_k=20)
    assert len(all_dominant) <= 10, "Should not exceed num_patterns"

    print(f"✓ Top-k={20}: returned {len(all_dominant)} patterns")

    print("\n✅ TEST 5 PASSED")


def test_reset():
    """Test reset functionality."""
    print("\n" + "="*70)
    print("TEST 6: RESET")
    print("="*70)

    identity = IdentityVector(num_patterns=5)

    # Build up some state
    for i in range(5):
        identity.reinforce(i % 5)

    assert not np.allclose(identity.state, 0), "State should be non-zero"
    assert identity.total_updates > 0, "Should have updates"

    print(f"✓ Before reset: {identity.total_updates} updates")
    print(f"  Max activation: {identity.state.max():.4f}")

    # Reset
    identity.reset()

    assert np.allclose(identity.state, 0), "State should be zero after reset"
    assert identity.total_updates == 0, "Update count should be zero"

    print(f"✓ After reset: {identity.total_updates} updates")
    print(f"  Max activation: {identity.state.max():.4f}")

    print("\n✅ TEST 6 PASSED")


def test_custom_strength():
    """Test reinforcement with custom strengths."""
    print("\n" + "="*70)
    print("TEST 7: CUSTOM REINFORCEMENT STRENGTH")
    print("="*70)

    identity = IdentityVector(num_patterns=5, update_strength=0.1)

    # Reinforce with different strengths
    identity.reinforce(0, strength=0.05)  # Weak
    identity.reinforce(1, strength=0.2)   # Medium
    identity.reinforce(2, strength=0.5)   # Strong

    print(f"✓ Weak (α=0.05): {identity.state[0]:.4f}")
    print(f"✓ Medium (α=0.2): {identity.state[1]:.4f}")
    print(f"✓ Strong (α=0.5): {identity.state[2]:.4f}")

    # Stronger reinforcements should produce larger values
    assert identity.state[0] < identity.state[1], "Weak < Medium"
    assert identity.state[1] < identity.state[2], "Medium < Strong"

    print(f"✓ Strength ordering preserved")

    print("\n✅ TEST 7 PASSED")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*70)
    print("TEST 8: EDGE CASES")
    print("="*70)

    identity = IdentityVector(num_patterns=3)

    # Test reinforcement at boundaries
    identity.reinforce(0)  # First pattern
    identity.reinforce(2)  # Last pattern

    assert identity.state[0] > 0, "First pattern not updated"
    assert identity.state[2] > 0, "Last pattern not updated"

    print(f"✓ Boundary indices work correctly")

    # Test decay on zero state
    identity2 = IdentityVector(num_patterns=5)
    identity2.decay()

    assert np.allclose(identity2.state, 0), "Decay on zeros should remain zero"

    print(f"✓ Decay on zero state stable")

    # Test get_state returns copy
    state1 = identity.get_state()
    state1[0] = 999  # Modify copy

    state2 = identity.get_state()
    assert state2[0] != 999, "get_state should return copy, not reference"

    print(f"✓ get_state returns copy")

    # Test dominant patterns on empty state
    identity3 = IdentityVector(num_patterns=5)
    dominant = identity3.get_dominant_patterns(top_k=3)

    assert len(dominant) == 0 or all(v < 1e-6 for _, v in dominant), "Empty state should have no dominant patterns"

    print(f"✓ Empty state handling correct")

    print("\n✅ TEST 8 PASSED")


def main():
    """Run all Identity Vector tests."""
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "IDENTITY VECTOR TEST SUITE" + " "*22 + "║")
    print("╚" + "="*68 + "╝")

    try:
        test_initialization()
        test_reinforcement()
        test_temporal_decay()
        test_sparsity()
        test_dominant_patterns()
        test_reset()
        test_custom_strength()
        test_edge_cases()

        print("\n" + "="*70)
        print("✅ ALL 8 IDENTITY VECTOR TESTS PASSED")
        print("="*70)
        print("\nValidated:")
        print("  ✓ Proper initialization with configurable parameters")
        print("  ✓ Soft reinforcement updates with saturation")
        print("  ✓ Temporal decay dynamics (exponential fade)")
        print("  ✓ Sparsity preservation (70-90% sparse)")
        print("  ✓ Dominant pattern tracking and sorting")
        print("  ✓ Reset functionality")
        print("  ✓ Custom reinforcement strengths")
        print("  ✓ Edge cases and boundary conditions")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
