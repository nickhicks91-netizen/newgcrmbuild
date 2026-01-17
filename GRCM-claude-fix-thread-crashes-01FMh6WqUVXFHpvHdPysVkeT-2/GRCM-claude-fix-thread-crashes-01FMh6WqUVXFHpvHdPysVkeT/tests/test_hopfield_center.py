"""
Validation Suite for Center-Only Hopfield Lattice

Tests all 5 metrics with center-only pseudo-inverse:
1. Self-healing > 70%
2. Memory recovery error < 0.1
3. Noise robustness > 80%
4. Topological persistence < 0.15
5. Pattern capacity ≥ 10

Expected: ALL TESTS PASS (5/5)

This is the CORRECT architecture: 27×27 weights for 27D center patterns.
Fixes the global/local mismatch that caused all previous implementations to fail.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/GRCM')

from grcm.echozero.identity.torsion_lattice.hopfield_center import HopfieldTorsionLatticeCenter


def test_self_healing():
    """
    TEST 1: Self-Healing > 70%

    Learn patterns, corrupt, verify energy descent recovers.
    """
    print("\n" + "="*70)
    print("TEST 1: SELF-HEALING (Target >70%)")
    print("="*70)

    lattice = HopfieldTorsionLatticeCenter(size=5)

    # Learn 3 patterns via center-only pseudo-inverse
    patterns = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.7, 0.7]),
    ]

    print(f"✓ Learning {len(patterns)} patterns via center-only pseudo-inverse...")
    result = lattice.learn_patterns_batch(patterns)

    if not result['success']:
        print(f"✗ Learning failed: {result.get('reason')}")
        return {'passed': False}

    # Write first pattern and relax
    lattice.write_vector(patterns[0], strength=0.5)
    print("✓ Relaxing to attractor...")
    for _ in range(100):
        lattice.step()

    stable_vec = lattice.read_vector()
    stable_energy = lattice.compute_energy()
    print(f"  Stable energy: {stable_energy:.4f}")

    # Corrupt with massive noise (center only - that's where the Hopfield network is!)
    print("✓ Injecting massive noise into center...")
    noise_phases = np.random.uniform(0, 2*np.pi, size=(3, 3, 3))
    noise = np.exp(1j * noise_phases)

    c = lattice.center_start
    current_center = lattice.state[c:c+3, c:c+3, c:c+3].copy()
    corrupted_center = (current_center + noise * 2.0)
    corrupted_center = corrupted_center / (np.abs(corrupted_center) + 1e-8)
    lattice.state[c:c+3, c:c+3, c:c+3] = corrupted_center

    corrupted_energy = lattice.compute_energy()
    print(f"  Corrupted energy: {corrupted_energy:.4f}")

    # Self-heal
    print("✓ Self-healing (200 energy descent steps)...")
    for step in range(200):
        lattice.step()
        if (step + 1) % 50 == 0:
            energy = lattice.compute_energy()
            print(f"  Step {step+1:3d}: energy = {energy:.4f}")

    # Measure recovery
    healed_vec = lattice.read_vector()

    # Vector correlation
    vector_corr = np.dot(stable_vec, healed_vec)

    healed_energy = lattice.compute_energy()
    print(f"\n✓ Vector correlation: {vector_corr:.4f}")
    print(f"✓ Final energy: {healed_energy:.4f}")

    passed = vector_corr > 0.70
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Self-healing = {vector_corr:.2%} (target >70%)")

    return {
        'vector_correlation': vector_corr,
        'passed': passed
    }


def test_memory_preservation():
    """
    TEST 2: Memory Preservation < 0.1 Error

    Learn patterns, retrieve each with noise.
    """
    print("\n" + "="*70)
    print("TEST 2: MEMORY PRESERVATION (Target error <0.1)")
    print("="*70)

    lattice = HopfieldTorsionLatticeCenter(size=5)

    # Create well-separated patterns (orthogonal)
    test_patterns = [
        np.array([1.0, 0.0]),       # 0°
        np.array([0.707, 0.707]),   # 45°
        np.array([0.0, 1.0]),       # 90°
        np.array([-0.707, 0.707]),  # 135°
        np.array([-1.0, 0.0]),      # 180°
        np.array([-0.707, -0.707]), # 225°
        np.array([0.0, -1.0]),      # 270°
        np.array([0.707, -0.707]),  # 315°
    ]

    print(f"\n✓ Learning {len(test_patterns)} well-separated patterns...")
    result = lattice.learn_patterns_batch(test_patterns)

    if not result['success']:
        print(f"✗ Learning failed: {result.get('reason')}")
        return {'passed': False}

    print(f"\n✓ Testing retrieval with noise...")
    errors = []

    for idx, pattern in enumerate(test_patterns):
        pattern_norm = pattern / np.linalg.norm(pattern)

        # Add noise
        noisy_pattern = pattern_norm + np.random.randn(2) * 0.1
        noisy_pattern = noisy_pattern / np.linalg.norm(noisy_pattern)

        # Write and relax
        lattice.write_vector(noisy_pattern, strength=0.5)
        for _ in range(100):
            lattice.step()

        # Read back
        recovered = lattice.read_vector()
        error = np.linalg.norm(recovered - pattern_norm)
        errors.append(error)

        if (idx + 1) % 2 == 0 or idx < 2:
            print(f"  Pattern {idx+1}: error = {error:.4f}")

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    errors_good = sum(1 for e in errors if e < 0.15)

    print(f"\n✓ Average error: {avg_error:.4f}")
    print(f"✓ Maximum error: {max_error:.4f}")
    print(f"✓ Patterns with error <0.15: {errors_good}/{len(test_patterns)}")

    passed = avg_error < 0.1
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Avg error = {avg_error:.4f} (target <0.1)")

    return {
        'errors': errors,
        'avg_error': avg_error,
        'max_error': max_error,
        'passed': passed
    }


def test_noise_robustness():
    """
    TEST 3: Noise Robustness > 80%

    Test recovery across noise levels 0 to 3.
    """
    print("\n" + "="*70)
    print("TEST 3: NOISE ROBUSTNESS (Target >80% recovery)")
    print("="*70)

    lattice = HopfieldTorsionLatticeCenter(size=5)

    # Learn 4 orthogonal patterns
    patterns = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]

    print("✓ Learning 4 orthogonal patterns...")
    result = lattice.learn_patterns_batch(patterns)

    if not result['success']:
        print(f"✗ Learning failed")
        return {'passed': False}

    # Test noise levels
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"\n✓ Testing noise robustness...")
    recovery_rates = []

    for noise in noise_levels:
        correct = 0
        total = len(patterns) * 5  # 5 trials per pattern

        for pattern in patterns:
            pattern_norm = pattern / np.linalg.norm(pattern)

            for trial in range(5):
                # Add noise
                noisy = pattern_norm + np.random.randn(2) * noise
                noisy = noisy / (np.linalg.norm(noisy) + 1e-8)

                # Write and relax
                lattice.write_vector(noisy, strength=0.5)
                for _ in range(100):
                    lattice.step()

                # Check recovery
                recovered = lattice.read_vector()
                error = np.linalg.norm(recovered - pattern_norm)

                if error < 0.3:  # Correct recovery threshold
                    correct += 1

        recovery_rate = correct / total
        recovery_rates.append(recovery_rate)
        print(f"  Noise σ={noise:.2f}: {recovery_rate*100:.1f}% recovery")

    # Check if maintains >80% up to σ=3.0
    passed = all(r > 0.80 for r in recovery_rates)

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Maintains >80% recovery across all noise levels")

    return {
        'noise_levels': noise_levels,
        'recovery_rates': recovery_rates,
        'passed': passed
    }


def test_topological_persistence():
    """
    TEST 4: Topological Persistence < 0.15 Error

    Test stability under rotation.
    """
    print("\n" + "="*70)
    print("TEST 4: TOPOLOGICAL PERSISTENCE (Target <0.15 error)")
    print("="*70)

    lattice = HopfieldTorsionLatticeCenter(size=5)

    # Learn single pattern for rotation test
    base_pattern = np.array([0.8, 0.6])
    base_pattern = base_pattern / np.linalg.norm(base_pattern)

    # Learn it
    print(f"✓ Learning base pattern...")
    result = lattice.learn_patterns_batch([base_pattern])

    if not result['success']:
        return {'passed': False}

    # Test rotation invariance
    rotation_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    errors = []

    print(f"\n✓ Testing rotation persistence...")

    for angle in rotation_angles:
        # Rotate pattern
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = rotation_matrix @ base_pattern

        # Write rotated
        lattice.write_vector(rotated, strength=0.5)

        # Relax
        for _ in range(100):
            lattice.step()

        # Read and rotate back
        recovered = lattice.read_vector()
        inverse_rotation = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        recovered_unrotated = inverse_rotation @ recovered

        # Error relative to base
        error = np.linalg.norm(recovered_unrotated - base_pattern)
        errors.append(error)

        print(f"  Rotation {angle*180/np.pi:5.1f}°: error = {error:.4f}")

    avg_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"\n✓ Average error: {avg_error:.4f}")
    print(f"✓ Maximum error: {max_error:.4f}")

    passed = avg_error < 0.15
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Avg error = {avg_error:.4f} (target <0.15)")

    return {
        'rotation_errors': errors,
        'avg_error': avg_error,
        'passed': passed
    }


def test_capacity():
    """
    TEST 5: Pattern Capacity ≥ 10 Patterns

    Store 15 patterns, verify good retrieval.
    """
    print("\n" + "="*70)
    print("TEST 5: PATTERN CAPACITY (Target ≥10 patterns)")
    print("="*70)

    lattice = HopfieldTorsionLatticeCenter(size=5, max_patterns=20)

    # Generate evenly-spaced patterns around circle
    num_patterns = 15
    patterns = []
    for i in range(num_patterns):
        angle = 2 * np.pi * i / num_patterns
        pattern = np.array([np.cos(angle), np.sin(angle)])
        patterns.append(pattern)

    print(f"✓ Learning {num_patterns} patterns via center-only pseudo-inverse...")

    # Check orthogonality before learning
    print(f"\n✓ Checking pattern orthogonality...")
    for i in range(min(3, len(patterns))):
        ortho_check = lattice.check_pattern_orthogonality(patterns[i])
        if i == 0:
            print(f"  Pattern 1: (first pattern, no conflicts)")
        else:
            print(f"  Pattern {i+1}: max overlap = {ortho_check['max_overlap']:.3f}")

    # Learn all patterns
    result = lattice.learn_patterns_batch(patterns)

    if not result['success']:
        print(f"✗ Learning failed: {result.get('reason')}")
        return {'passed': False}

    # Test retrieval
    print(f"\n✓ Testing retrieval accuracy...")
    errors = []

    for i, pattern in enumerate(patterns):
        # Add small noise
        noisy = pattern + np.random.randn(2) * 0.05
        noisy = noisy / np.linalg.norm(noisy)

        # Write and retrieve
        lattice.write_vector(noisy, strength=0.5)
        for _ in range(100):
            lattice.step()

        recovered = lattice.read_vector()
        pattern_norm = pattern / np.linalg.norm(pattern)
        error = np.linalg.norm(recovered - pattern_norm)
        errors.append(error)

        if (i + 1) % 3 == 0 or i < 3:
            print(f"  Pattern {i+1}: error = {error:.4f}")

    avg_error = np.mean(errors)
    errors_good = sum(1 for e in errors if e < 0.15)

    print(f"\n✓ Average retrieval error: {avg_error:.4f}")
    print(f"✓ Patterns with error <0.15: {errors_good}/{num_patterns}")

    passed = errors_good >= 10
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: {errors_good} patterns with good retrieval (target ≥10)")

    return {
        'num_patterns': num_patterns,
        'avg_error': avg_error,
        'errors_good': errors_good,
        'passed': passed
    }


def main():
    """Run all center-only pseudo-inverse validation tests"""
    print("╔" + "="*68 + "╗")
    print("║" + " "*4 + "CENTER-ONLY HOPFIELD LATTICE - VALIDATION SUITE" + " "*15 + "║")
    print("║" + " "*8 + "27×27 Weights for 27D Center (CORRECT Architecture)" + " "*9 + "║")
    print("╚" + "="*68 + "╝")

    results = {}

    # Run all tests
    results['self_healing'] = test_self_healing()
    results['memory'] = test_memory_preservation()
    results['noise'] = test_noise_robustness()
    results['topological'] = test_topological_persistence()
    results['capacity'] = test_capacity()

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    tests_passed = sum(1 for r in results.values() if r.get('passed', False))
    total_tests = len(results)

    print(f"\nTests Passed: {tests_passed}/{total_tests}")
    print(f"  1. Self-Healing (>70%):           {'✓ PASS' if results['self_healing']['passed'] else '✗ FAIL'}")
    print(f"  2. Memory Preservation (<0.1):    {'✓ PASS' if results['memory']['passed'] else '✗ FAIL'}")
    print(f"  3. Noise Robustness (>80%):       {'✓ PASS' if results['noise']['passed'] else '✗ FAIL'}")
    print(f"  4. Topological Persistence:       {'✓ PASS' if results['topological']['passed'] else '✗ FAIL'}")
    print(f"  5. Pattern Capacity (≥10):        {'✓ PASS' if results['capacity']['passed'] else '✗ FAIL'}")

    print(f"\nKey Metrics:")
    print(f"  Self-healing correlation: {results['self_healing']['vector_correlation']*100:.1f}%")
    print(f"  Memory avg error: {results['memory']['avg_error']:.4f}")
    print(f"  Noise recovery (min): {min(results['noise']['recovery_rates'])*100:.1f}%")
    print(f"  Rotation avg error: {results['topological']['avg_error']:.4f}")
    print(f"  Pattern capacity: {results['capacity']['errors_good']} patterns")

    print("\n" + "="*70)
    if tests_passed == total_tests:
        print("✅ ALL TESTS PASSED - Center-only architecture validated!")
        print("\nThe lattice now:")
        print("  ✓ Self-heals from corruption (>70% recovery)")
        print("  ✓ Preserves identity information (<0.1 error)")
        print("  ✓ Robust to noise (>80% recovery)")
        print("  ✓ Topologically persistent (<0.15 error)")
        print("  ✓ Stores multiple patterns (≥10 capacity)")
        print("\n🎉 PRODUCTION READY FOR ECHOZERO INTEGRATION")
    else:
        print(f"⚠ {total_tests - tests_passed} test(s) failed")
        print("\nComparison across all implementations:")
        print("  Hebbian v1:        1/5 passing (pattern interference)")
        print("  Pseudo-inv v2:     1/5 passing (scalar/vector mismatch)")
        print("  Phase-only v3:     1/5 passing (circular variable issue)")
        print("  Complex v4:        1/5 passing (global/local mismatch)")
        print(f"  Center-only v5:    {tests_passed}/5 passing (CORRECT architecture)")

    print("="*70)

    return results


if __name__ == '__main__':
    main()
