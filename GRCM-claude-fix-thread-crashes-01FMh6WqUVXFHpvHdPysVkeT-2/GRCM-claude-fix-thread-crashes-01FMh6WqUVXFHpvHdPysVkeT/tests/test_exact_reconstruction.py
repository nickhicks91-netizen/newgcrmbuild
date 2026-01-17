"""
Exact Reconstruction Test Suite (6 tests)

Validates replay buffer for perfect state reconstruction:
- Exact state retrieval
- Sequence reconstruction
- Temporal coherence tracking
- Discontinuity detection
- Buffer overflow handling
- Metadata preservation
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/GRCM')

from grcm.echozero.state.replay_buffer import ReplayBuffer


def test_exact_reconstruction():
    """Test 1: Perfect reconstruction of stored states."""
    print("\n" + "="*70)
    print("TEST 1: EXACT RECONSTRUCTION")
    print("="*70)

    buffer = ReplayBuffer(dim=64, capacity=10)

    # Store states
    states = [np.random.randn(64) for _ in range(5)]

    for state in states:
        buffer.push(state, torsion=1.0, phase_grad=0.2)

    # Verify exact reconstruction
    for i in range(1, len(states) + 1):
        reconstructed = buffer.reconstruct_exact(idx=-i)
        original = states[-i]

        error = np.linalg.norm(reconstructed - original)

        print(f"  State -{i}: error={error:.10f}")

        assert error < 1e-6, f"Reconstruction not exact: {error}"

    print(f"\n✓ Perfect reconstruction (error < 1e-6)")

    print("\n✅ TEST 1 PASSED")


def test_sequence_reconstruction():
    """Test 2: Temporal sequence reconstruction."""
    print("\n" + "="*70)
    print("TEST 2: SEQUENCE RECONSTRUCTION")
    print("="*70)

    buffer = ReplayBuffer(dim=64, capacity=100)

    # Create temporal sequence
    for t in range(20):
        state = np.random.randn(64)
        buffer.push(state, torsion=float(t), phase_grad=0.1)

    # Reconstruct sequence
    sequence = buffer.reconstruct_sequence(k=10)

    print(f"✓ Sequence length: {len(sequence)}")
    print(f"✓ Last torsion: {sequence[-1]['torsion']}")
    print(f"✓ First (in window) torsion: {sequence[0]['torsion']}")

    assert len(sequence) == 10
    assert sequence[-1]['torsion'] == 19.0  # Most recent
    assert sequence[0]['torsion'] == 10.0   # Oldest in window

    print("\n✅ TEST 2 PASSED")


def test_temporal_coherence():
    """Test 3: Temporal coherence measurement."""
    print("\n" + "="*70)
    print("TEST 3: TEMPORAL COHERENCE")
    print("="*70)

    buffer = ReplayBuffer(dim=64, capacity=100)

    # Test 1: Smooth sequence (high coherence)
    print("\n  Smooth sequence:")
    base = np.random.randn(64)
    for t in range(20):
        state = base + np.random.randn(64) * 0.01  # Small noise
        buffer.push(state)

    coherence_smooth = buffer.compute_temporal_coherence(k=10)
    print(f"    Coherence: {coherence_smooth:.4f}")

    # Test 2: Chaotic sequence (low coherence)
    print("\n  Chaotic sequence:")
    buffer.clear()
    for t in range(20):
        state = np.random.randn(64)  # Large random jumps
        buffer.push(state)

    coherence_chaotic = buffer.compute_temporal_coherence(k=10)
    print(f"    Coherence: {coherence_chaotic:.4f}")

    print(f"\n✓ Smooth coherence ({coherence_smooth:.4f}) > Chaotic ({coherence_chaotic:.4f})")

    assert coherence_smooth > coherence_chaotic

    print("\n✅ TEST 3 PASSED")


def test_discontinuity_detection():
    """Test 4: Torsion spike detection."""
    print("\n" + "="*70)
    print("TEST 4: DISCONTINUITY DETECTION")
    print("="*70)

    buffer = ReplayBuffer(dim=64, capacity=100)

    # Create sequence with spikes
    spike_steps = [5, 12, 18]

    for t in range(20):
        state = np.random.randn(64)
        torsion = 10.0 if t in spike_steps else 1.0
        buffer.push(state, torsion=torsion)

    # Detect spikes
    detected = buffer.detect_discontinuities(threshold=5.0, window=20)

    print(f"✓ Expected spikes at: {spike_steps}")
    print(f"✓ Detected spikes at: {detected}")

    # Should detect all spikes
    for step in spike_steps:
        assert step in detected, f"Missed spike at step {step}"

    print("\n✅ TEST 4 PASSED")


def test_buffer_overflow():
    """Test 5: Buffer handles overflow correctly (FIFO)."""
    print("\n" + "="*70)
    print("TEST 5: BUFFER OVERFLOW (FIFO)")
    print("="*70)

    buffer = ReplayBuffer(dim=64, capacity=10)

    # Push more than capacity
    for t in range(15):
        state = np.ones(64) * t  # Identifiable states
        buffer.push(state, torsion=float(t))

    # Should only keep last 10
    assert len(buffer) == 10

    # Oldest should be step 5 (15 - 10)
    oldest = buffer.buffer[0]
    assert oldest['step'] == 5

    # Newest should be step 14
    newest = buffer.buffer[-1]
    assert newest['step'] == 14

    print(f"✓ Buffer size: {len(buffer)} (capacity: {buffer.capacity})")
    print(f"✓ Oldest step: {oldest['step']}")
    print(f"✓ Newest step: {newest['step']}")

    print("\n✅ TEST 5 PASSED")


def test_metadata_preservation():
    """Test 6: Metadata is preserved correctly."""
    print("\n" + "="*70)
    print("TEST 6: METADATA PRESERVATION")
    print("="*70)

    buffer = ReplayBuffer(dim=64, capacity=100)

    # Push with custom metadata
    metadata = {
        'label': 'important_state',
        'confidence': 0.95,
        'attractor_idx': 3,
    }

    state = np.random.randn(64)
    buffer.push(state, torsion=2.5, phase_grad=0.8, metadata=metadata)

    # Retrieve and check
    entry = buffer.last()

    assert entry['torsion'] == 2.5
    assert entry['phase_grad'] == 0.8
    assert entry['metadata']['label'] == 'important_state'
    assert entry['metadata']['confidence'] == 0.95
    assert entry['metadata']['attractor_idx'] == 3

    print(f"✓ Torsion: {entry['torsion']}")
    print(f"✓ Phase grad: {entry['phase_grad']}")
    print(f"✓ Metadata: {entry['metadata']}")

    print("\n✅ TEST 6 PASSED")


def main():
    """Run all exact reconstruction tests."""
    print("╔" + "="*68 + "╗")
    print("║" + " "*17 + "EXACT RECONSTRUCTION TEST SUITE" + " "*19 + "║")
    print("╚" + "="*68 + "╝")

    try:
        test_exact_reconstruction()
        test_sequence_reconstruction()
        test_temporal_coherence()
        test_discontinuity_detection()
        test_buffer_overflow()
        test_metadata_preservation()

        print("\n" + "="*70)
        print("✅ ALL 6 EXACT RECONSTRUCTION TESTS PASSED")
        print("="*70)
        print("\nValidated:")
        print("  ✓ Perfect state reconstruction (error < 1e-6)")
        print("  ✓ Temporal sequence retrieval")
        print("  ✓ Coherence measurement (smooth vs chaotic)")
        print("  ✓ Torsion spike detection")
        print("  ✓ FIFO buffer overflow handling")
        print("  ✓ Metadata preservation")

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
