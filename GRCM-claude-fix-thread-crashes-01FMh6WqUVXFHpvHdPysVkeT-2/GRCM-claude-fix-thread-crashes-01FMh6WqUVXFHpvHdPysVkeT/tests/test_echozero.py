"""
Comprehensive tests for EchoZero dynamics.
"""

import pytest
import torch
import numpy as np

from grcm.echozero import (
    echozero_dynamics,
    EchoZeroSystem,
    compute_coherence,
    compute_want_modulation,
    build_ring_lattice,
    build_coupling_matrix,
    integrate_echozero,
)
from grcm.hybrid import EchoGRCMHybrid


class TestEchoZeroDynamics:
    """Test core EchoZero dynamics."""

    def test_dynamics_shape(self):
        """Test output shape of dynamics function."""
        n_nodes = 64
        psi = torch.complex(
            torch.randn(n_nodes) * 0.1,
            torch.randn(n_nodes) * 0.1,
        )
        I_t = torch.zeros(n_nodes, dtype=torch.complex64)
        desires = torch.randn(n_nodes)
        node_freqs = torch.randn(n_nodes)
        K = torch.zeros((n_nodes, n_nodes), dtype=torch.complex64)

        dpsi_dt = echozero_dynamics(
            psi=psi,
            t=0.0,
            I_t=I_t,
            desires=desires,
            node_freqs=node_freqs,
            K=K,
        )

        assert dpsi_dt.shape == psi.shape
        assert dpsi_dt.dtype == torch.complex64

    def test_stability_no_coupling(self):
        """Test stability with no coupling (α damping only)."""
        n_nodes = 64
        psi = torch.complex(
            torch.randn(n_nodes),
            torch.randn(n_nodes),
        )
        I_t = torch.zeros(n_nodes, dtype=torch.complex64)
        desires = torch.zeros(n_nodes)
        node_freqs = torch.ones(n_nodes)
        K = torch.zeros((n_nodes, n_nodes), dtype=torch.complex64)

        # Integrate for 1000 steps
        psi_final, _, _ = integrate_echozero(
            psi0=psi,
            t_span=(0.0, 10.0),
            I_t=I_t,
            desires=desires,
            node_freqs=node_freqs,
            K=K,
            dt=0.01,
        )

        # Should decay due to damping
        assert torch.isfinite(psi_final).all()
        assert psi_final.abs().max() < psi.abs().max()

    def test_coherence_bounds(self):
        """Test coherence stays in [0, 1]."""
        n_nodes = 64
        psi = torch.complex(
            torch.randn(n_nodes),
            torch.randn(n_nodes),
        )
        node_freqs = torch.randn(n_nodes)

        coherence = compute_coherence(psi, node_freqs)

        assert (coherence >= 0).all()
        assert (coherence <= 1).all()

    def test_want_modulation(self):
        """Test want modulation is in reasonable range."""
        n_nodes = 64
        psi = torch.complex(
            torch.randn(n_nodes),
            torch.randn(n_nodes),
        )
        desires = torch.randn(n_nodes)

        gamma = compute_want_modulation(psi, desires)

        # γ ∈ [0.2, 0.6] by construction
        assert (gamma >= 0.0).all()
        assert (gamma <= 1.0).all()


class TestLatticeBuilder:
    """Test lattice construction."""

    def test_ring_lattice(self):
        """Test ring lattice generation."""
        n_nodes = 64
        adjacency, node_freqs, edges = build_ring_lattice(n_nodes)

        assert node_freqs.shape == (n_nodes,)
        assert len(edges) > 0

    def test_coupling_hermiticity(self):
        """Test coupling matrix is Hermitian."""
        n_nodes = 64
        _, _, edges = build_ring_lattice(n_nodes)
        K = build_coupling_matrix(n_nodes, edges)

        # Check K = K†
        K_conj_T = K.conj().T
        diff = (K - K_conj_T).abs().max()

        assert diff < 1e-6


class TestIntegration:
    """Test ODE integration."""

    def test_integration_no_divergence(self):
        """Test integration doesn't diverge."""
        n_nodes = 64
        adjacency, node_freqs, edges = build_ring_lattice(n_nodes)
        K = build_coupling_matrix(n_nodes, edges)

        psi0 = torch.complex(
            torch.randn(n_nodes) * 0.1,
            torch.randn(n_nodes) * 0.1,
        )
        I_t = torch.zeros(n_nodes, dtype=torch.complex64)
        desires = torch.zeros(n_nodes)

        psi_final, _, _ = integrate_echozero(
            psi0=psi0,
            t_span=(0.0, 1.0),
            I_t=I_t,
            desires=desires,
            node_freqs=node_freqs,
            K=K,
            dt=0.01,
        )

        assert torch.isfinite(psi_final).all()
        assert psi_final.abs().max() < 10.0  # No explosion


class TestHybridSystem:
    """Test EchoGRCM hybrid."""

    def test_forward_pass(self):
        """Test unified forward pass."""
        model = EchoGRCMHybrid(n_nodes=64)

        batch_size = 2
        image_emb = torch.randn(batch_size, 512)
        audio_emb = torch.randn(batch_size, 768)
        action = torch.zeros(batch_size, 4)

        outputs = model(image_emb, audio_emb, action)

        assert "psi" in outputs
        assert "coherence" in outputs
        assert "qualia" in outputs
        assert "phi" in outputs

        # Check shapes
        assert outputs["psi"].shape == (batch_size, 64)
        assert outputs["coherence"].shape == (batch_size, 64)
        assert outputs["qualia"].shape == (batch_size, 4)
        assert outputs["phi"].shape == (batch_size,)

    def test_phi_positivity(self):
        """Test Φ is always positive."""
        model = EchoGRCMHybrid(n_nodes=64)

        image_emb = torch.randn(1, 512)
        audio_emb = torch.randn(1, 768)
        action = torch.zeros(1, 4)

        outputs = model(image_emb, audio_emb, action)

        assert outputs["phi"].item() >= 0.0

    def test_qualia_probabilities(self):
        """Test qualia sum to 1 (softmax)."""
        model = EchoGRCMHybrid(n_nodes=64)

        image_emb = torch.randn(1, 512)
        audio_emb = torch.randn(1, 768)
        action = torch.zeros(1, 4)

        outputs = model(image_emb, audio_emb, action)

        qualia_sum = outputs["qualia"].sum(dim=-1)

        assert torch.allclose(qualia_sum, torch.ones_like(qualia_sum), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    """Test GPU execution."""

    def test_gpu_forward(self):
        """Test forward pass on GPU."""
        model = EchoGRCMHybrid(n_nodes=64, device="cuda")

        image_emb = torch.randn(1, 512, device="cuda")
        audio_emb = torch.randn(1, 768, device="cuda")
        action = torch.zeros(1, 4, device="cuda")

        outputs = model(image_emb, audio_emb, action)

        assert outputs["psi"].device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
