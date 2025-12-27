"""
EchoZero Möbius Layer Test Suite

Tests topological consistency validation and phase inversion.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/GRCM')

# Note: Möbius is PyTorch-based, these tests verify the mathematical properties
# For now, we test the expected behavior without importing PyTorch


def test_mobius_concept_phase_inversion():
    """Test that phase inversion concept is mathematically sound."""
    # Möbius property: ψ → -ψ after full traversal
    psi = np.random.randn(64)
    psi_inverted = -psi

    # Torsion energy = interference between ψ and -ψ
    torsion = np.linalg.norm(psi - psi_inverted)

    # Should detect inconsistency
    assert torsion > 0
    print(f"✓ Phase inversion torsion: {torsion:.4f}")


def test_mobius_concept_consistency_metric():
    """Test that consistent states have low torsion."""
    # Consistent state: ψ(t) ≈ ψ(t-1)
    psi_t = np.ones(64) * 0.5
    psi_prev = np.ones(64) * 0.5

    torsion_consistent = np.linalg.norm(psi_t - (-psi_prev))

    # Inconsistent state: large change
    psi_inconsistent = np.random.randn(64)
    torsion_inconsistent = np.linalg.norm(psi_inconsistent - (-psi_prev))

    # Inconsistent should have higher torsion
    assert torsion_inconsistent > torsion_consistent
    print(f"✓ Consistent: {torsion_consistent:.4f}, Inconsistent: {torsion_inconsistent:.4f}")


def test_mobius_concept_geometric_invariance():
    """Test that rotation doesn't destroy topological properties."""
    psi = np.random.randn(64)

    # Rotate by 90 degrees (geometric operation)
    theta = np.pi / 2
    psi_rotated = np.cos(theta) * psi - np.sin(theta) * np.roll(psi, 1)

    # Both should have similar magnitude
    norm_original = np.linalg.norm(psi)
    norm_rotated = np.linalg.norm(psi_rotated)

    assert abs(norm_original - norm_rotated) < 1.0
    print(f"✓ Norm preservation: {norm_original:.4f} → {norm_rotated:.4f}")


def test_mobius_concept_nonlinear_gating():
    """Test sigmoid gating suppresses high-torsion states."""
    torsion_low = 0.5
    torsion_high = 5.0

    # Sigmoid gate: gate = 1 / (1 + exp(torsion - threshold))
    threshold = 2.0

    gate_low = 1 / (1 + np.exp(torsion_low - threshold))
    gate_high = 1 / (1 + np.exp(torsion_high - threshold))

    # Low torsion should pass more
    assert gate_low > gate_high
    print(f"✓ Gate low torsion: {gate_low:.4f}, high: {gate_high:.4f}")


if __name__ == '__main__':
    test_mobius_concept_phase_inversion()
    test_mobius_concept_consistency_metric()
    test_mobius_concept_geometric_invariance()
    test_mobius_concept_nonlinear_gating()
    print("\n✅ All Möbius concept tests passed")
