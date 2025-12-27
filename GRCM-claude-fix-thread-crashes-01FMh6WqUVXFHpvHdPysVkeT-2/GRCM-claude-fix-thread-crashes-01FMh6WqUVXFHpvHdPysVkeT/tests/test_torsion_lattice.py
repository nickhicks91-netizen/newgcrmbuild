"""
Comprehensive tests for 3D Torsion Memory Lattice

Tests cover:
- Stability and self-healing
- Write and read operations
- Möbius gating
- Background synchronization
- Integration with EchoGRCM hybrid
"""

import pytest
import numpy as np
import torch

from grcm.echozero.identity.torsion_lattice import (
    TorsionLattice3D,
    LatticeUpdater,
    read_identity,
    write_identity,
    read_identity_with_stats,
    compute_identity_distance,
    safe_write_with_validation,
    incremental_write,
)


class TestTorsionLattice3D:
    """Test core lattice functionality"""

    def test_initialization(self):
        """Test lattice initializes correctly"""
        L = TorsionLattice3D(size=5)

        assert L.size == 5
        assert L.theta.shape == (5, 5, 5)
        assert L.m == 1.0
        assert len(L.neighbors) == 26

    def test_checkerboard_initialization(self):
        """Test checkerboard pattern initialization"""
        L = TorsionLattice3D(size=3)

        # Check alternating pattern
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if (i + j + k) % 2 == 0:
                        assert L.theta[i, j, k] == 0.0
                    else:
                        assert L.theta[i, j, k] == np.pi

    def test_stability(self):
        """Test lattice remains stable over 2000 steps"""
        L = TorsionLattice3D(size=5)

        for _ in range(2000):
            L.step()

        # Magnitude should decay but remain positive
        assert L.m > 0
        assert L.m < 1.0

        # Phases should remain finite
        assert np.isfinite(L.theta).all()

        # Phases should remain in [0, 2π)
        assert np.all(L.theta >= 0)
        assert np.all(L.theta < 2 * np.pi)

    def test_self_healing(self):
        """Test lattice self-heals from noise"""
        L = TorsionLattice3D(size=5, coupling=0.8)

        # Measure initial energy
        initial_energy = L.get_energy()

        # Add random noise
        L.theta += np.random.randn(5, 5, 5) * 0.5
        L.theta = np.mod(L.theta, 2 * np.pi)

        # Measure perturbed energy
        perturbed_energy = L.get_energy()
        assert perturbed_energy > initial_energy  # Energy increased

        # Relax for 100 steps
        for _ in range(100):
            L.step()

        # Energy should decrease (healing)
        healed_energy = L.get_energy()
        assert healed_energy < perturbed_energy

    def test_write_and_read(self):
        """Test write and read operations"""
        L = TorsionLattice3D(size=5)

        # Write identity vector
        vec = np.array([1.0, 0.0])
        L.write_vector(vec, strength=0.05)

        # Read back
        out = L.read_vector()

        # Should be aligned (real component positive)
        assert out[0] > 0.1

    def test_write_strength_constraint(self):
        """Test write strength is applied correctly"""
        L = TorsionLattice3D(size=5)

        # Store initial state
        theta_before = L.theta.copy()

        # Write with different strengths
        vec = np.array([0.8, 0.6])

        L.write_vector(vec, strength=0.01)
        theta_weak = L.theta.copy()

        L.theta = theta_before.copy()
        L.write_vector(vec, strength=0.05)
        theta_strong = L.theta.copy()

        # Stronger write should cause bigger change
        weak_diff = np.abs(theta_weak - theta_before).mean()
        strong_diff = np.abs(theta_strong - theta_before).mean()

        assert strong_diff > weak_diff

    def test_magnetization(self):
        """Test order parameter (magnetization) computation"""
        L = TorsionLattice3D(size=5)

        # Checkerboard should have some magnetization
        mag_init = L.get_magnetization()
        assert 0 <= mag_init <= 1

        # Uniform state should have high magnetization
        L.theta[:] = 0.0
        mag_uniform = L.get_magnetization()
        assert mag_uniform > 0.9

        # Random state should have low magnetization
        L.theta = np.random.rand(5, 5, 5) * 2 * np.pi
        mag_random = L.get_magnetization()
        assert mag_random < 0.5

    def test_reset(self):
        """Test lattice reset"""
        L = TorsionLattice3D(size=5)

        # Perturb lattice
        L.theta += np.random.randn(5, 5, 5)
        L.m = 0.5

        # Reset
        L.reset()

        # Should be back to checkerboard
        assert L.m == 1.0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if (i + j + k) % 2 == 0:
                        assert L.theta[i, j, k] == 0.0
                    else:
                        assert L.theta[i, j, k] == np.pi


class TestLatticeUpdater:
    """Test background updater (slow loop manager)"""

    def test_initialization(self):
        """Test updater initializes correctly"""
        L = TorsionLattice3D(size=5)
        updater = LatticeUpdater(
            lattice=L,
            write_strength=0.03,
            sync_interval=300,
            torsion_threshold=0.2,
        )

        assert updater.step_count == 0
        assert updater.total_syncs == 0
        assert updater.sync_interval == 300

    def test_sync_interval_gating(self):
        """Test sync only happens at correct intervals"""
        L = TorsionLattice3D(size=5)
        updater = LatticeUpdater(lattice=L, sync_interval=10)

        echo_state = np.array([0.5, 0.5])
        torsion = 0.1  # Low torsion

        # Steps 1-9 should not sync
        for _ in range(9):
            result = updater.maybe_sync(echo_state, torsion)
            assert result["synced"] is False
            assert result["reason"] == "not_interval"

        # Step 10 should sync
        result = updater.maybe_sync(echo_state, torsion)
        assert result["synced"] is True
        assert updater.total_syncs == 1

    def test_mobius_gating(self):
        """Test Möbius gate rejects high torsion"""
        L = TorsionLattice3D(size=5)
        updater = LatticeUpdater(
            lattice=L,
            sync_interval=1,  # Sync every step
            torsion_threshold=0.2,
        )

        echo_state = np.array([0.5, 0.5])

        # Low torsion: should sync
        result = updater.maybe_sync(echo_state, torsion_score=0.1)
        assert result["synced"] is True

        # High torsion: should reject
        result = updater.maybe_sync(echo_state, torsion_score=0.5)
        assert result["synced"] is False
        assert result["reason"] == "high_torsion"
        assert updater.rejected_syncs == 1

    def test_sync_stats(self):
        """Test sync statistics tracking"""
        L = TorsionLattice3D(size=5)
        updater = LatticeUpdater(lattice=L, sync_interval=5)

        echo_state = np.array([0.7, 0.3])

        # Perform multiple syncs with mixed torsion
        for i in range(15):
            torsion = 0.1 if i % 2 == 0 else 0.5  # Alternate low/high
            updater.maybe_sync(echo_state, torsion)

        stats = updater.get_stats()

        assert stats["step_count"] == 15
        assert stats["total_syncs"] >= 0
        assert stats["rejected_syncs"] >= 0
        assert 0 <= stats["acceptance_rate"] <= 1

    def test_force_sync(self):
        """Test forced sync (bypass interval and torsion checks)"""
        L = TorsionLattice3D(size=5)
        updater = LatticeUpdater(lattice=L, sync_interval=1000)

        echo_state = np.array([0.8, 0.2])

        # Force sync even though interval not reached
        result = updater.force_sync(echo_state)

        assert result["synced"] is True
        assert result["forced"] is True
        assert updater.total_syncs == 1

    def test_reset(self):
        """Test updater reset"""
        L = TorsionLattice3D(size=5)
        updater = LatticeUpdater(lattice=L, sync_interval=1)

        # Perform some syncs
        for _ in range(5):
            updater.maybe_sync(np.array([0.5, 0.5]), 0.1)

        # Reset updater only
        updater.reset()

        assert updater.step_count == 0
        assert updater.total_syncs == 0
        assert updater.rejected_syncs == 0


class TestReadWriteUtilities:
    """Test readout and writeback utilities"""

    def test_read_identity(self):
        """Test basic identity readout"""
        L = TorsionLattice3D(size=5)
        L.theta[:] = np.pi / 4  # Uniform angle

        identity = read_identity(L)

        assert identity.shape == (2,)
        assert np.isfinite(identity).all()

    def test_read_identity_with_stats(self):
        """Test identity readout with statistics"""
        L = TorsionLattice3D(size=5)

        stats = read_identity_with_stats(L)

        assert "identity" in stats
        assert "magnetization" in stats
        assert "energy" in stats
        assert "magnitude" in stats
        assert stats["size"] == 5

    def test_compute_identity_distance(self):
        """Test identity vector distance metric"""
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        vec_c = np.array([1.0, 0.0])

        # Orthogonal vectors
        dist_ab = compute_identity_distance(vec_a, vec_b)
        assert 0 < dist_ab < 1

        # Identical vectors
        dist_ac = compute_identity_distance(vec_a, vec_c)
        assert dist_ac < 0.01

        # Opposite vectors
        vec_d = np.array([-1.0, 0.0])
        dist_ad = compute_identity_distance(vec_a, vec_d)
        assert dist_ad > 0.9

    def test_safe_write_with_validation(self):
        """Test safe write with Möbius gating"""
        L = TorsionLattice3D(size=5)
        vec = np.array([0.8, 0.6])

        # Low torsion: should write
        result = safe_write_with_validation(
            L, vec, torsion_score=0.1, torsion_threshold=0.2
        )
        assert result["written"] is True

        # High torsion: should reject
        result = safe_write_with_validation(
            L, vec, torsion_score=0.5, torsion_threshold=0.2
        )
        assert result["written"] is False
        assert result["reason"] == "high_torsion"

        # Invalid vector shape: should reject
        bad_vec = np.array([1, 2, 3])
        result = safe_write_with_validation(L, bad_vec)
        assert result["written"] is False

    def test_incremental_write(self):
        """Test incremental write across multiple steps"""
        L = TorsionLattice3D(size=5)
        target = np.array([1.0, 0.0])

        initial_vec = L.read_vector()

        # Incremental write
        incremental_write(L, target, num_steps=10, total_strength=0.05)

        final_vec = L.read_vector()

        # Should move toward target
        dist_before = compute_identity_distance(initial_vec, target)
        dist_after = compute_identity_distance(final_vec, target)

        assert dist_after < dist_before


@pytest.mark.integration
class TestHybridIntegration:
    """Test integration with EchoGRCM hybrid system"""

    def test_hybrid_with_torsion_lattice(self):
        """Test EchoGRCM hybrid with torsion lattice enabled"""
        from grcm.hybrid import EchoGRCMHybrid

        # Create hybrid with torsion lattice
        model = EchoGRCMHybrid(
            n_nodes=32,
            enable_torsion_lattice=True,
            lattice_sync_interval=10,
            device="cpu",
        )

        assert model.enable_torsion_lattice is True
        assert model.torsion_lattice is not None
        assert model.lattice_updater is not None

    def test_hybrid_forward_with_lattice(self):
        """Test forward pass with lattice sync"""
        from grcm.hybrid import EchoGRCMHybrid

        model = EchoGRCMHybrid(
            n_nodes=32,
            enable_torsion_lattice=True,
            lattice_sync_interval=5,
            device="cpu",
        )

        # Prepare inputs
        image_emb = torch.randn(1, 512)
        audio_emb = torch.randn(1, 768)
        action = torch.zeros(1, 4)

        # Run forward passes
        for i in range(12):
            outputs = model(image_emb, audio_emb, action)

            # Check outputs exist
            assert "psi" in outputs
            assert "torsion_score" in outputs
            assert "lattice_sync_info" in outputs

            # Every 5 steps, sync should attempt
            if (i + 1) % 5 == 0:
                sync_info = outputs["lattice_sync_info"]
                assert sync_info is not None

    def test_hybrid_without_torsion_lattice(self):
        """Test hybrid works with lattice disabled (backward compat)"""
        from grcm.hybrid import EchoGRCMHybrid

        model = EchoGRCMHybrid(
            n_nodes=32,
            enable_torsion_lattice=False,
            device="cpu",
        )

        assert model.enable_torsion_lattice is False
        assert model.torsion_lattice is None

        # Should still work
        image_emb = torch.randn(1, 512)
        audio_emb = torch.randn(1, 768)
        action = torch.zeros(1, 4)

        outputs = model(image_emb, audio_emb, action)

        assert outputs["lattice_sync_info"] is None
        assert "torsion_score" in outputs  # Möbius still active


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
