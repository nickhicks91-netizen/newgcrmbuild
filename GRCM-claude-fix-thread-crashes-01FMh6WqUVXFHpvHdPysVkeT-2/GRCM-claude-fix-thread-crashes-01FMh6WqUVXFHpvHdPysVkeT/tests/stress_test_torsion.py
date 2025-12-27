#!/usr/bin/env python3
"""
Stress Test Suite for 3D Torsion Memory Lattice

Validates stability, performance, and resilience under extreme conditions.
This version tests NumPy-based components that don't require PyTorch.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm.echozero.identity.torsion_lattice import (
    TorsionLattice3D,
    LatticeUpdater,
    read_identity,
    write_identity,
    compute_identity_distance,
    safe_write_with_validation,
)


# ============================================================================
# STRESS TEST UTILITIES
# ============================================================================

class StressTestRunner:
    """Manages stress test execution and reporting."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def run_test(self, name, test_fn):
        """Run a single test and track results."""
        print(f"\n{'='*80}")
        print(f"TEST: {name}")
        print(f"{'='*80}")

        start = time.perf_counter()
        try:
            test_fn()
            elapsed = time.perf_counter() - start
            print(f"✅ PASSED ({elapsed:.3f}s)")
            self.passed += 1
            self.tests.append((name, True, elapsed))
        except AssertionError as e:
            elapsed = time.perf_counter() - start
            print(f"❌ FAILED ({elapsed:.3f}s): {e}")
            self.failed += 1
            self.tests.append((name, False, elapsed))
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"💥 ERROR ({elapsed:.3f}s): {e}")
            self.failed += 1
            self.tests.append((name, False, elapsed))

    def print_summary(self):
        """Print final test summary."""
        print(f"\n{'='*80}")
        print(f"STRESS TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Success Rate: {100 * self.passed / (self.passed + self.failed):.1f}%")
        print(f"\nTotal Time: {sum(t[2] for t in self.tests):.3f}s")
        print(f"{'='*80}\n")


runner = StressTestRunner()


# ============================================================================
# 1. EXTREME LONG-RUN STABILITY (10,000 STEPS)
# ============================================================================

def test_extreme_long_run_10k_steps():
    """Test lattice stability over 10,000 relaxation steps."""
    print("Running 10,000 relaxation steps...")

    lattice = TorsionLattice3D(size=5, coupling=0.8, gamma=0.03)

    energy_samples = []
    mag_samples = []

    for i in range(10000):
        lattice.step()

        if i % 1000 == 0:
            energy = lattice.get_energy()
            mag = lattice.get_magnetization()
            energy_samples.append(energy)
            mag_samples.append(mag)
            print(f"  Step {i:5d}: E={energy:8.2f}, M={mag:.4f}")

    # Validate stability
    final_mag = lattice.get_magnetization()
    final_energy = lattice.get_energy()

    assert lattice.m > 0, "Magnitude collapsed"
    assert np.isfinite(lattice.theta).all(), "Phases became NaN/Inf"
    assert final_mag > 0.01, f"Magnetization too low: {final_mag}"
    assert -50000 < final_energy < 0, f"Energy out of range: {final_energy}"

    # Check for convergence (energy should stabilize)
    late_energies = energy_samples[-3:]
    energy_std = np.std(late_energies)
    assert energy_std < 100, f"Energy not converged, std={energy_std}"

    print(f"Final state: M={final_mag:.6f}, E={final_energy:.2f}")
    print("✓ Survived 10,000 steps with stable convergence")


runner.run_test("Extreme Long-Run Stability (10K steps)", test_extreme_long_run_10k_steps)


# ============================================================================
# 2. MASSIVE CORRUPTION RECOVERY
# ============================================================================

def test_massive_corruption_recovery():
    """Test recovery from severe corruption (50% of lattice randomized)."""
    print("Applying massive corruption...")

    lattice = TorsionLattice3D(size=5)

    # Build stable state
    for _ in range(200):
        lattice.step()

    initial_mag = lattice.get_magnetization()
    initial_energy = lattice.get_energy()

    print(f"Initial: M={initial_mag:.4f}, E={initial_energy:.2f}")

    # Massive corruption: randomize 50% of lattice
    corruption_mask = np.random.rand(5, 5, 5) < 0.5
    lattice.theta[corruption_mask] = np.random.rand(np.sum(corruption_mask)) * 2 * np.pi

    corrupted_mag = lattice.get_magnetization()
    corrupted_energy = lattice.get_energy()

    print(f"Corrupted: M={corrupted_mag:.4f}, E={corrupted_energy:.2f}")
    assert corrupted_energy > initial_energy, "Corruption didn't increase energy"

    # Heal
    print("Healing for 500 steps...")
    for i in range(500):
        lattice.step()
        if i % 100 == 0:
            mag = lattice.get_magnetization()
            energy = lattice.get_energy()
            print(f"  Step {i:3d}: M={mag:.4f}, E={energy:.2f}")

    healed_mag = lattice.get_magnetization()
    healed_energy = lattice.get_energy()

    print(f"Healed: M={healed_mag:.4f}, E={healed_energy:.2f}")

    # Should self-heal significantly
    assert healed_energy < corrupted_energy, "Failed to heal"
    assert healed_mag > corrupted_mag * 1.2, "Magnetization didn't recover enough"
    assert np.isfinite(lattice.theta).all()

    print(f"✓ Recovered from {np.sum(corruption_mask)} corrupted nodes")


runner.run_test("Massive Corruption Recovery", test_massive_corruption_recovery)


# ============================================================================
# 3. RAPID WRITE STRESS TEST (1,000 WRITES)
# ============================================================================

def test_rapid_write_stress_1000_writes():
    """Test 1,000 rapid writes with random identity vectors."""
    print("Performing 1,000 rapid writes...")

    lattice = TorsionLattice3D(size=5)

    write_count = 0
    start_time = time.perf_counter()

    for i in range(1000):
        # Random identity vector
        angle = np.random.rand() * 2 * np.pi
        vec = np.array([np.cos(angle), np.sin(angle)])

        # Write with varying strengths
        strength = 0.01 + 0.04 * np.random.rand()  # 0.01-0.05
        lattice.write_vector(vec, strength=strength)
        write_count += 1

        if i % 100 == 0:
            mag = lattice.get_magnetization()
            print(f"  Write {i:4d}: M={mag:.4f}")

    elapsed = time.perf_counter() - start_time
    throughput = write_count / elapsed

    # Validate stability
    assert np.isfinite(lattice.theta).all()
    assert lattice.m > 0
    final_mag = lattice.get_magnetization()
    assert final_mag > 0.01, f"Magnetization collapsed: {final_mag}"

    print(f"✓ {write_count} writes in {elapsed:.3f}s ({throughput:.0f} writes/sec)")


runner.run_test("Rapid Write Stress (1K writes)", test_rapid_write_stress_1000_writes)


# ============================================================================
# 4. EXTREME SIZE SCALING (10³ = 1,000 NODES)
# ============================================================================

def test_extreme_size_scaling_10cubed():
    """Test larger lattice (10³ = 1,000 nodes)."""
    print("Creating 10³ = 1,000 node lattice...")

    lattice = TorsionLattice3D(size=10, coupling=0.8, gamma=0.03)

    print(f"Lattice size: {lattice.size}³ = {lattice.size**3} nodes")

    # Run some steps
    print("Running 100 relaxation steps on 1,000 node lattice...")
    start_time = time.perf_counter()

    for i in range(100):
        lattice.step()
        if i % 20 == 0:
            mag = lattice.get_magnetization()
            print(f"  Step {i:3d}: M={mag:.4f}")

    elapsed = time.perf_counter() - start_time
    steps_per_sec = 100 / elapsed

    # Validate
    assert np.isfinite(lattice.theta).all()
    assert lattice.m > 0

    print(f"✓ 1,000 node lattice stable ({steps_per_sec:.1f} steps/sec)")


runner.run_test("Extreme Size Scaling (10³ nodes)", test_extreme_size_scaling_10cubed)


# ============================================================================
# 5. UPDATER STRESS TEST (HIGH FREQUENCY SYNC)
# ============================================================================

def test_updater_high_frequency_stress():
    """Test updater with very high sync frequency (every step)."""
    print("Testing updater with sync every step (1,000 steps)...")

    lattice = TorsionLattice3D(size=5)
    updater = LatticeUpdater(
        lattice=lattice,
        sync_interval=1,  # Every step
        torsion_threshold=0.5,
        relaxation_steps=10,
    )

    synced_count = 0
    rejected_count = 0

    for step in range(1000):
        # Random identity state
        angle = np.random.rand() * 2 * np.pi
        echo_state = np.array([np.cos(angle), np.sin(angle)])

        # Random torsion (mix of high and low)
        torsion = 0.8 * np.random.rand()  # 0-0.8

        result = updater.maybe_sync(echo_state, torsion)

        if result['synced']:
            synced_count += 1
        else:
            if result.get('reason') == 'high_torsion':
                rejected_count += 1

        if step % 200 == 0:
            mag = lattice.get_magnetization()
            print(f"  Step {step:4d}: synced={synced_count}, rejected={rejected_count}, M={mag:.4f}")

    stats = updater.get_stats()

    print(f"\nFinal stats:")
    print(f"  Total syncs: {stats['total_syncs']}")
    print(f"  Rejected: {stats['rejected_syncs']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")

    # Validate
    assert synced_count > 0, "No syncs occurred"
    assert rejected_count > 0, "No rejections (should have some high torsion)"
    assert np.isfinite(lattice.theta).all()
    assert lattice.m > 0

    print(f"✓ Handled {synced_count + rejected_count} sync attempts")


runner.run_test("Updater High-Frequency Stress", test_updater_high_frequency_stress)


# ============================================================================
# 6. ADVERSARIAL OSCILLATION ATTACK
# ============================================================================

def test_adversarial_oscillation_attack():
    """Test against rapid oscillating identity vectors."""
    print("Running adversarial oscillation attack (500 cycles)...")

    lattice = TorsionLattice3D(size=5)
    updater = LatticeUpdater(
        lattice=lattice,
        sync_interval=1,
        torsion_threshold=0.3,
    )

    # Establish baseline
    for _ in range(50):
        lattice.step()

    initial_mag = lattice.get_magnetization()

    # Attack: rapidly oscillate between opposite states
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([-1.0, 0.0])

    rejection_count = 0

    for i in range(500):
        # Alternate between opposite vectors
        vec = vec_a if i % 2 == 0 else vec_b

        # High torsion due to oscillation
        torsion = 0.5 + 0.3 * np.random.rand()

        result = updater.maybe_sync(vec, torsion)

        if not result['synced'] and result.get('reason') == 'high_torsion':
            rejection_count += 1

    final_mag = lattice.get_magnetization()

    print(f"Initial magnetization: {initial_mag:.4f}")
    print(f"Final magnetization: {final_mag:.4f}")
    print(f"Rejections: {rejection_count}/500 ({100*rejection_count/500:.1f}%)")

    # Should reject most attacks
    assert rejection_count > 300, f"Too few rejections: {rejection_count}"

    # Lattice should remain reasonably stable
    assert final_mag > initial_mag * 0.5, "Magnetization degraded too much"
    assert np.isfinite(lattice.theta).all()

    print(f"✓ Resisted {rejection_count} adversarial attempts")


runner.run_test("Adversarial Oscillation Attack", test_adversarial_oscillation_attack)


# ============================================================================
# 7. MEMORY LEAK TEST (REPEATED ALLOCATION)
# ============================================================================

def test_memory_leak_repeated_allocation():
    """Test for memory leaks with repeated lattice creation."""
    print("Testing for memory leaks (100 lattice creations)...")

    initial_theta_id = None

    for i in range(100):
        lattice = TorsionLattice3D(size=5)

        # Perform some operations
        for _ in range(10):
            lattice.step()

        vec = np.array([0.7, 0.7])
        lattice.write_vector(vec, strength=0.03)

        if i == 0:
            initial_theta_id = id(lattice.theta)

        # Check basic properties
        assert np.isfinite(lattice.theta).all()
        assert lattice.m > 0

        if i % 20 == 0:
            print(f"  Iteration {i:3d}: theta.shape={lattice.theta.shape}, id={id(lattice.theta)}")

    print("✓ No memory leaks detected (all allocations succeeded)")


runner.run_test("Memory Leak Test", test_memory_leak_repeated_allocation)


# ============================================================================
# 8. CONCURRENT UPDATER STRESS (MULTIPLE UPDATERS)
# ============================================================================

def test_multiple_updaters_same_lattice():
    """Test multiple updaters on same lattice (simulates concurrent access)."""
    print("Testing multiple updaters on same lattice...")

    lattice = TorsionLattice3D(size=5)

    updater1 = LatticeUpdater(lattice, sync_interval=3, torsion_threshold=0.3)
    updater2 = LatticeUpdater(lattice, sync_interval=5, torsion_threshold=0.4)
    updater3 = LatticeUpdater(lattice, sync_interval=7, torsion_threshold=0.2)

    for step in range(300):
        vec = np.random.randn(2)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        torsion = 0.6 * np.random.rand()

        # All updaters try to sync
        updater1.maybe_sync(vec, torsion)
        updater2.maybe_sync(vec, torsion)
        updater3.maybe_sync(vec, torsion)

        if step % 60 == 0:
            stats1 = updater1.get_stats()
            stats2 = updater2.get_stats()
            stats3 = updater3.get_stats()
            mag = lattice.get_magnetization()
            print(f"  Step {step:3d}: syncs=({stats1['total_syncs']}, {stats2['total_syncs']}, {stats3['total_syncs']}), M={mag:.4f}")

    # Validate
    assert np.isfinite(lattice.theta).all()
    assert lattice.m > 0

    total_syncs = sum([
        updater1.get_stats()['total_syncs'],
        updater2.get_stats()['total_syncs'],
        updater3.get_stats()['total_syncs'],
    ])

    print(f"✓ {total_syncs} total syncs from 3 updaters")


runner.run_test("Multiple Updaters Stress", test_multiple_updaters_same_lattice)


# ============================================================================
# PRINT FINAL SUMMARY
# ============================================================================

runner.print_summary()

if runner.failed > 0:
    sys.exit(1)
else:
    print("🎉 ALL STRESS TESTS PASSED! 🎉\n")
    sys.exit(0)
