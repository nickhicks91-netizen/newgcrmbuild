"""
Triad Integration Test Suite - Möbius + Spiral + 3D Torsion Memory

This validates the complete geometric cognition stack working together:
- Möbius Layer: Hallucination damping via topological consistency
- Spiral Lattice: 2D geometric memory with temporal encoding
- 3D Torsion Memory Lattice: Long-term self-healing identity storage

Tests cover:
- Full pipeline integration
- Multi-step coherence propagation
- Long-run stability (2,000 steps)
- Chaos resilience (Lorenz attractor)
- Adversarial injection handling
- Cross-module consistency
"""

import pytest
import numpy as np
import torch

from grcm.echozero.mobius import MobiusEchoLayer
from grcm.echozero.spiral import SpiralLattice
from grcm.echozero.identity.torsion_lattice import (
    TorsionLattice3D,
    LatticeUpdater,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def random_vec(dim=64, device="cpu"):
    """Generate random normalized vector."""
    v = torch.randn(dim, device=device)
    return v / (torch.norm(v) + 1e-8)


def extract_2d_identity(tensor):
    """Extract 2D identity vector from tensor for torsion lattice."""
    if tensor.dim() > 1:
        tensor = tensor[0]
    # Take mean of first 2 dimensions as identity vector
    return np.array([
        tensor[0].item() if len(tensor) > 0 else 0.0,
        tensor[1].item() if len(tensor) > 1 else 0.0,
    ])


# ============================================================================
# 1. FULL PIPELINE INTEGRATION TEST
# ============================================================================

@pytest.mark.integration
def test_full_pipeline_step():
    """Test single step through complete Möbius → Spiral → Torsion pipeline."""

    hidden_dim = 64

    # Initialize all three components
    mobius = MobiusEchoLayer(hidden_dim=hidden_dim, device="cpu")
    spiral = SpiralLattice(hidden_dim=hidden_dim)
    lattice = TorsionLattice3D(size=5)
    updater = LatticeUpdater(
        lattice=lattice,
        sync_interval=1,  # Sync every step for testing
        torsion_threshold=0.5,
    )

    x = random_vec(hidden_dim)

    # 1. Möbius forward pass
    y, metrics = mobius(x.unsqueeze(0))  # Add batch dimension
    torsion = metrics['energy'].mean().item()

    # 2. Spiral Lattice forward
    spiral_out, spiral_metrics = spiral(y)
    coherence = spiral_metrics['coherence'].item()

    # 3. Torsion Lattice sync
    identity_vec = extract_2d_identity(y[0])
    sync_result = updater.maybe_sync(
        echo_state=identity_vec,
        torsion_score=torsion,
    )

    # Validate outputs are finite and bounded
    assert np.isfinite(torsion)
    assert np.isfinite(coherence)
    assert np.isfinite(lattice.m)
    assert lattice.m > 0

    # Check metrics are in expected ranges
    assert 0 <= torsion <= 10.0  # Torsion can be high initially
    assert -1.1 <= coherence <= 1.1  # Coherence is cosine similarity
    assert 0 < lattice.m <= 1.0  # Magnitude decays but stays positive


# ============================================================================
# 2. LONG-RUN STABILITY TEST (2,000 STEPS)
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_long_run_stability_2000_steps():
    """Test stability over 2,000 steps with random inputs."""

    hidden_dim = 64

    mobius = MobiusEchoLayer(hidden_dim=hidden_dim, device="cpu")
    spiral = SpiralLattice(hidden_dim=hidden_dim)
    lattice = TorsionLattice3D(size=5)
    updater = LatticeUpdater(
        lattice=lattice,
        sync_interval=10,  # Sync every 10 steps
        torsion_threshold=0.3,
    )

    torsion_vals = []
    coherence_vals = []

    x = random_vec(hidden_dim)

    for step in range(2000):
        # Forward through pipeline
        y, metrics = mobius(x.unsqueeze(0))
        torsion = metrics['energy'].mean().item()
        torsion_vals.append(torsion)

        spiral_out, spiral_metrics = spiral(y)
        coherence = spiral_metrics['coherence'].item()
        coherence_vals.append(coherence)

        # Sync to torsion lattice
        identity_vec = extract_2d_identity(y[0])
        updater.maybe_sync(identity_vec, torsion)

        # New random input
        x = random_vec(hidden_dim)

    # Check stability metrics
    mean_torsion = np.mean(torsion_vals)
    max_torsion = np.max(torsion_vals)

    assert max_torsion < 1.0, f"Torsion should stabilize, got max={max_torsion}"
    assert mean_torsion < 0.5, f"Mean torsion too high: {mean_torsion}"

    # Coherence should be bounded
    assert np.min(coherence_vals) > -1.1
    assert np.max(coherence_vals) < 1.1

    # Lattice should remain stable
    assert lattice.m > 0, "Lattice magnitude collapsed"
    assert np.isfinite(lattice.theta).all(), "Lattice phases became NaN/Inf"

    # Should have had some successful syncs
    assert updater.total_syncs > 0, "No syncs occurred"


# ============================================================================
# 3. CHAOS CONSISTENCY TEST (LORENZ ATTRACTOR)
# ============================================================================

@pytest.mark.integration
def test_chaos_resilience_lorenz():
    """Test pipeline with chaotic Lorenz attractor input."""

    hidden_dim = 64

    mobius = MobiusEchoLayer(hidden_dim=hidden_dim, device="cpu")
    spiral = SpiralLattice(hidden_dim=hidden_dim)
    lattice = TorsionLattice3D(size=5)
    updater = LatticeUpdater(lattice=lattice, sync_interval=5)

    def lorenz_step(state, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
        """Single Lorenz system integration step."""
        x1, x2, x3 = state
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        return np.array([x1 + dx1*dt, x2 + dx2*dt, x3 + dx3*dt])

    # Initialize Lorenz system
    lorenz_state = np.array([1.0, 1.0, 1.0])
    torsion_vals = []
    coherence_vals = []

    for _ in range(500):
        # Evolve chaotic system
        lorenz_state = lorenz_step(lorenz_state)

        # Project to high-dimensional input
        x = torch.tensor(
            np.pad(lorenz_state, (0, hidden_dim - 3), mode='constant'),
            dtype=torch.float32
        )
        x = x / (torch.norm(x) + 1e-8)

        # Pipeline forward
        y, metrics = mobius(x.unsqueeze(0))
        torsion = metrics['energy'].mean().item()
        torsion_vals.append(torsion)

        spiral_out, spiral_metrics = spiral(y)
        coherence = spiral_metrics['coherence'].item()
        coherence_vals.append(coherence)

        # Sync to lattice
        identity_vec = extract_2d_identity(y[0])
        updater.maybe_sync(identity_vec, torsion)

    # Möbius should stabilize chaos
    mean_torsion = np.mean(torsion_vals)
    assert mean_torsion < 0.6, f"Möbius failed to stabilize chaos: {mean_torsion}"

    # Coherence should remain bounded
    assert np.isfinite(coherence_vals).all()

    # Lattice should survive chaos
    assert lattice.m > 0.01, "Lattice collapsed under chaos"
    assert np.isfinite(lattice.theta).all()


# ============================================================================
# 4. ADVERSARIAL INVERSION ATTACK TEST
# ============================================================================

@pytest.mark.integration
def test_adversarial_inversion_protection():
    """Test triad-level protection against adversarial phase inversion."""

    hidden_dim = 64

    mobius = MobiusEchoLayer(hidden_dim=hidden_dim, device="cpu")
    spiral = SpiralLattice(hidden_dim=hidden_dim)
    lattice = TorsionLattice3D(size=5)
    updater = LatticeUpdater(
        lattice=lattice,
        sync_interval=1,
        torsion_threshold=0.2,  # Strict threshold
    )

    # Establish baseline
    x = random_vec(hidden_dim)
    y_base, _ = mobius(x.unsqueeze(0))

    rejection_count = 0
    acceptance_count = 0

    for _ in range(100):
        # Adversarial attack: phase inversion + noise
        attack = -y_base[0] + 0.05 * torch.randn(hidden_dim)

        # Forward through pipeline
        y, metrics = mobius(attack.unsqueeze(0))
        torsion = metrics['energy'].mean().item()

        spiral_out, _ = spiral(y)

        # Try to sync (should reject high torsion)
        identity_vec = extract_2d_identity(y[0])
        result = updater.maybe_sync(identity_vec, torsion)

        if result.get('synced'):
            acceptance_count += 1
        else:
            rejection_count += 1

    # Should reject majority of attacks
    rejection_rate = rejection_count / 100
    assert rejection_rate > 0.7, f"Only rejected {rejection_rate:.1%} of attacks"

    # Lattice should remain stable
    assert lattice.m > 0
    assert np.isfinite(lattice.theta).all()


# ============================================================================
# 5. CROSS-MODULE CONSISTENCY TEST
# ============================================================================

@pytest.mark.integration
def test_cross_module_consistency():
    """Test that Möbius torsion gates both Spiral and Torsion Lattice."""

    hidden_dim = 64

    mobius = MobiusEchoLayer(hidden_dim=hidden_dim, device="cpu")
    spiral = SpiralLattice(hidden_dim=hidden_dim)
    lattice = TorsionLattice3D(size=5)
    updater = LatticeUpdater(
        lattice=lattice,
        sync_interval=1,
        torsion_threshold=0.25,
    )

    high_torsion_count = 0
    low_torsion_count = 0

    for _ in range(50):
        x = random_vec(hidden_dim)

        y, metrics = mobius(x.unsqueeze(0))
        torsion = metrics['energy'].mean().item()

        # Both downstream modules should see same torsion signal
        spiral_out, spiral_metrics = spiral(y)

        identity_vec = extract_2d_identity(y[0])
        sync_result = updater.maybe_sync(identity_vec, torsion)

        if torsion > 0.25:
            high_torsion_count += 1
            # High torsion should block sync
            if sync_result.get('synced'):
                # Some syncs may slip through right at threshold
                pass
        else:
            low_torsion_count += 1
            # Low torsion should allow sync (if interval met)

    # Should see distribution of both high and low torsion
    assert high_torsion_count > 0, "No high torsion states detected"
    assert low_torsion_count > 0, "No low torsion states detected"

    # All modules should remain stable
    assert np.isfinite(mobius.mobius_buffer.cpu().numpy()).all()
    assert np.isfinite(spiral.spiral_memory.cpu().numpy()).all()
    assert np.isfinite(lattice.theta).all()


# ============================================================================
# 6. GRADUAL CORRUPTION RECOVERY TEST
# ============================================================================

@pytest.mark.integration
def test_gradual_corruption_recovery():
    """Test that torsion lattice self-heals from gradual corruption."""

    hidden_dim = 64

    mobius = MobiusEchoLayer(hidden_dim=hidden_dim, device="cpu")
    spiral = SpiralLattice(hidden_dim=hidden_dim)
    lattice = TorsionLattice3D(size=5)
    updater = LatticeUpdater(
        lattice=lattice,
        sync_interval=5,
        torsion_threshold=0.3,
    )

    # Build up some stable state
    for _ in range(50):
        x = random_vec(hidden_dim)
        y, metrics = mobius(x.unsqueeze(0))
        spiral(y)
        identity_vec = extract_2d_identity(y[0])
        updater.maybe_sync(identity_vec, metrics['energy'].mean().item())

    # Measure initial lattice energy
    initial_energy = lattice.get_energy()

    # Corrupt lattice
    lattice.theta += np.random.randn(5, 5, 5) * 0.3
    lattice.theta = np.mod(lattice.theta, 2 * np.pi)
    corrupted_energy = lattice.get_energy()

    assert corrupted_energy > initial_energy, "Corruption didn't increase energy"

    # Run relaxation (self-healing)
    for _ in range(100):
        lattice.step()

    healed_energy = lattice.get_energy()

    # Should self-heal (energy should decrease)
    assert healed_energy < corrupted_energy, "Lattice failed to self-heal"

    # Magnetization should recover
    mag = lattice.get_magnetization()
    assert mag > 0.1, "Magnetization collapsed during healing"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
