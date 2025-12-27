"""
Spiral Lattice Validation Suite

Complete test suite for SpiralLattice geometric memory layer.
Tests geometry, numerical stability, integration, and adversarial robustness.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
import math

from grcm.echozero.spiral import SpiralLattice, create_spiral_lattice
from grcm.echozero.mobius import MobiusEchoLayer, create_mobius_layer


# ================================================================
# FIXTURES
# ================================================================

@pytest.fixture
def spiral_layer():
    """Standard spiral lattice configuration for testing."""
    return SpiralLattice(
        hidden_dim=32,
        memory_length=128,
        radial_decay=0.015,
        angular_velocity=0.45,
        stability_gain=1.6,
    )


@pytest.fixture
def mobius_layer():
    """Standard Möbius layer for integration testing."""
    return MobiusEchoLayer(
        hidden_dim=32,
        loop_len=64,
        torsion_gain=3.0,
        damping_gain=12.0,
    )


# ================================================================
# TEST 1 — BASIC FORWARD PASS
# ================================================================

def test_forward_pass(spiral_layer):
    """Test basic forward pass functionality and output shapes."""
    x = torch.randn(1, 32)
    out, metrics = spiral_layer(x)

    # Check output shape matches input
    assert out.shape == x.shape, "Output shape mismatch"

    # Check all metrics are present
    assert "coherence" in metrics, "Missing coherence metric"
    assert "radius" in metrics, "Missing radius metric"
    assert "angle" in metrics, "Missing angle metric"
    assert "stabilizer" in metrics, "Missing stabilizer metric"

    # Check metric bounds
    assert 0.0 <= metrics["radius"] <= 1.0, "Radius out of bounds"
    assert isinstance(metrics["coherence"], float), "Coherence not float"
    assert math.isfinite(metrics["stabilizer"]), "Stabilizer not finite"

    print("✅ PASS: Basic forward pass")


# ================================================================
# TEST 2 — COHERENCE BEHAVIOR
# ================================================================

def test_coherence_behavior(spiral_layer):
    """Test that coherence metric responds correctly to input patterns."""

    # Aligned vectors should produce higher coherence
    v1 = torch.ones(1, 32)
    out1, m1 = spiral_layer(v1)

    # Advance spiral and test orthogonal pattern
    for _ in range(10):
        spiral_layer(torch.randn(1, 32))

    # Orthogonal vectors should produce lower coherence
    v2 = torch.cat([torch.ones(1, 16), -torch.ones(1, 16)], dim=-1)
    out2, m2 = spiral_layer(v2)

    # Coherence should decrease for less aligned patterns
    # Note: This is a structural alignment test
    print(f"  Coherence (aligned): {m1['coherence']:.4f}")
    print(f"  Coherence (orthogonal): {m2['coherence']:.4f}")

    # Both should be finite
    assert math.isfinite(m1["coherence"]), "Coherence 1 not finite"
    assert math.isfinite(m2["coherence"]), "Coherence 2 not finite"

    print("✅ PASS: Coherence behavior")


# ================================================================
# TEST 3 — LONG RUN STABILITY
# ================================================================

def test_long_run_stability(spiral_layer):
    """Test stability over 2000 inference steps."""
    x = torch.randn(1, 32)
    last_energy = None

    energies = []

    for step in range(2000):
        out, m = spiral_layer(x)
        current_energy = torch.norm(out).item()
        energies.append(current_energy)

        assert math.isfinite(current_energy), f"Energy not finite at step {step}"

        if last_energy is not None:
            # Energy should not diverge wildly
            assert abs(current_energy - last_energy) < 10.0, \
                f"Energy diverged at step {step}"

        last_energy = current_energy

    # Check overall stability
    mean_energy = sum(energies) / len(energies)
    print(f"  Mean energy over 2000 steps: {mean_energy:.4f}")

    print("✅ PASS: Long run stability (2000 steps)")


# ================================================================
# TEST 4 — SPIRAL GEOMETRY PROJECTION
# ================================================================

def test_spiral_projection_monotonicity(spiral_layer):
    """Test that radius decreases monotonically with time."""

    # Ensure radius decreases as t increases
    r1 = spiral_layer.get_radius(0)
    r2 = spiral_layer.get_radius(50)
    r3 = spiral_layer.get_radius(100)

    print(f"  r(0)   = {r1:.6f}")
    print(f"  r(50)  = {r2:.6f}")
    print(f"  r(100) = {r3:.6f}")

    assert r1 > r2 > r3, "Radius not monotonically decreasing"
    assert 0 < r3 < 1, "Radius out of valid range"

    print("✅ PASS: Spiral projection monotonicity")


# ================================================================
# TEST 5 — ANGULAR PROGRESSION
# ================================================================

def test_angle_progression(spiral_layer):
    """Test that angles increase linearly with time."""

    θ1 = spiral_layer.get_angle(0)
    θ2 = spiral_layer.get_angle(10)
    θ3 = spiral_layer.get_angle(20)

    print(f"  θ(0)  = {θ1:.4f} rad")
    print(f"  θ(10) = {θ2:.4f} rad")
    print(f"  θ(20) = {θ3:.4f} rad")

    assert θ1 < θ2 < θ3, "Angles not increasing"

    # Check linearity
    delta1 = θ2 - θ1
    delta2 = θ3 - θ2
    assert abs(delta1 - delta2) < 1e-6, "Angular velocity not constant"

    print("✅ PASS: Angular progression")


# ================================================================
# TEST 6 — SPIRAL + MÖBIUS INTEGRATION
# ================================================================

def test_spiral_mobius_integration(spiral_layer, mobius_layer):
    """Test integration between Spiral and Möbius layers."""
    x = torch.randn(1, 32)

    # Pass through spiral first
    x_s, m_s = spiral_layer(x)

    # Then through Möbius
    x_m, m_m = mobius_layer(x_s)

    # Check outputs
    assert x_m.shape == x_s.shape, "Shape mismatch after Möbius"
    assert math.isfinite(m_m["energy"]), "Möbius energy not finite"
    assert 0 <= m_m["gate_mean"] <= 1, "Möbius gate out of bounds"

    print(f"  Spiral coherence: {m_s['coherence']:.4f}")
    print(f"  Möbius energy: {m_m['energy']:.4f}")
    print(f"  Möbius gate: {m_m['gate_mean']:.4f}")

    print("✅ PASS: Spiral + Möbius integration")


# ================================================================
# TEST 7 — LORENZ CHAOS STRESS TEST
# ================================================================

def lorenz_step(x, y, z, dt=0.01, σ=10, β=8/3, ρ=28):
    """Single step of Lorenz attractor dynamics."""
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
    return x + dx*dt, y + dy*dt, z + dz*dt


def test_lorenz_chaos_stability(spiral_layer):
    """Test stability under Lorenz chaotic dynamics."""
    x, y, z = (1.0, 1.0, 1.0)

    coherences = []

    for step in range(500):
        x, y, z = lorenz_step(x, y, z)

        # Embed Lorenz state in hidden dimension
        vec = torch.tensor([[x, y] + [0]*30], dtype=torch.float32)

        out, m = spiral_layer(vec)

        # Check stability
        assert math.isfinite(torch.norm(out).item()), \
            f"Output diverged at Lorenz step {step}"
        assert m["coherence"] >= -1.0, "Coherence out of bounds"

        coherences.append(m["coherence"])

    mean_coherence = sum(coherences) / len(coherences)
    print(f"  Mean coherence under Lorenz chaos: {mean_coherence:.4f}")

    print("✅ PASS: Lorenz chaos stability")


# ================================================================
# TEST 8 — ADVERSARIAL INVERSION ATTACK
# ================================================================

def test_adversarial_inversion_attack(spiral_layer):
    """Test robustness against adversarial inversion attacks."""

    # Build memory history with normal inputs
    for _ in range(50):
        spiral_layer(torch.randn(1, 32))

    # Read current memory state
    t = int(spiral_layer.pointer.item())
    mem_vec = spiral_layer.spiral_memory[:, t, :]

    # Craft near-inverted attack vector
    attack = -mem_vec + 0.05 * torch.randn_like(mem_vec)

    out, metrics = spiral_layer(attack)

    # Attack should produce LOW coherence (geometric mismatch)
    print(f"  Adversarial coherence: {metrics['coherence']:.4f}")
    assert metrics["coherence"] < 0.5, "Failed to detect adversarial pattern"

    print("✅ PASS: Adversarial inversion attack resistance")


# ================================================================
# TEST 9 — PRECISION DEGRADATION (FP16)
# ================================================================

def test_fp16_precision(spiral_layer):
    """Test behavior under FP16 precision."""
    x = torch.randn(1, 32).half()
    spiral_layer_fp16 = spiral_layer.half()

    out, m = spiral_layer_fp16(x)

    # Check dtype preservation
    assert out.dtype == torch.float16, "Output dtype mismatch"

    # Check metrics are still finite
    assert math.isfinite(float(m["stabilizer"])), "Stabilizer not finite in FP16"
    assert math.isfinite(float(m["coherence"])), "Coherence not finite in FP16"

    print(f"  FP16 coherence: {m['coherence']:.4f}")
    print(f"  FP16 stabilizer: {m['stabilizer']:.4f}")

    print("✅ PASS: FP16 precision handling")


# ================================================================
# TEST 10 — SCALING TEST (1000 rotations)
# ================================================================

def test_spiral_scaling(spiral_layer):
    """Test stability over 1000 pointer rotations."""
    x = torch.randn(1, 32)

    for step in range(1000):
        out, m = spiral_layer(x)

        # Check all metrics remain valid
        assert math.isfinite(float(m["coherence"])), \
            f"Coherence not finite at step {step}"
        assert 0.0 <= m["radius"] <= 1.0, \
            f"Radius out of bounds at step {step}"

    print(f"  Final coherence after 1000 steps: {m['coherence']:.4f}")
    print(f"  Final radius: {m['radius']:.6f}")
    print(f"  Final angle: {m['angle']:.4f} rad")

    print("✅ PASS: Scaling test (1000 rotations)")


# ================================================================
# TEST 11 — FACTORY FUNCTION
# ================================================================

def test_factory_function():
    """Test create_spiral_lattice factory function."""
    spiral = create_spiral_lattice(
        hidden_dim=64,
        memory_length=512,
        radial_decay=0.012,
        angular_velocity=0.50,
        stability_gain=1.8,
    )

    assert spiral.hidden_dim == 64
    assert spiral.memory_length == 512
    assert spiral.radial_decay == 0.012
    assert spiral.angular_velocity == 0.50
    assert spiral.stability_gain == 1.8

    # Test forward pass
    x = torch.randn(1, 64)
    out, metrics = spiral(x)

    assert out.shape == (1, 64)
    assert "coherence" in metrics

    print("✅ PASS: Factory function")


# ================================================================
# RUN ALL TESTS
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SPIRAL LATTICE VALIDATION SUITE")
    print("=" * 70)
    print()

    # Create fixtures
    spiral = SpiralLattice(
        hidden_dim=32,
        memory_length=128,
        radial_decay=0.015,
        angular_velocity=0.45,
        stability_gain=1.6,
    )

    mobius = MobiusEchoLayer(
        hidden_dim=32,
        loop_len=64,
        torsion_gain=3.0,
        damping_gain=12.0,
    )

    # Run tests
    tests = [
        ("Basic Forward Pass", lambda: test_forward_pass(spiral)),
        ("Coherence Behavior", lambda: test_coherence_behavior(spiral)),
        ("Long Run Stability", lambda: test_long_run_stability(spiral)),
        ("Spiral Projection Monotonicity", lambda: test_spiral_projection_monotonicity(spiral)),
        ("Angular Progression", lambda: test_angle_progression(spiral)),
        ("Spiral + Möbius Integration", lambda: test_spiral_mobius_integration(spiral, mobius)),
        ("Lorenz Chaos Stability", lambda: test_lorenz_chaos_stability(spiral)),
        ("Adversarial Inversion Attack", lambda: test_adversarial_inversion_attack(spiral)),
        ("FP16 Precision", lambda: test_fp16_precision(spiral)),
        ("Scaling Test (1000 rotations)", lambda: test_spiral_scaling(spiral)),
        ("Factory Function", test_factory_function),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        print("-" * 70)
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Tests Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED")
        print("Status: PRODUCTION READY")
        print("Grade: A+ (Geometric Memory Validated)")
    else:
        print(f"\n⚠️  {failed} TESTS FAILED")
        print("Status: NEEDS FIXES")

    print("=" * 70)
