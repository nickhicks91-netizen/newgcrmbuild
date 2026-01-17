"""
Test Suite for Memory Router Integration Layer

Tests routing coordination across:
- Hierarchical replay (temporal memory)
- Semantic buffer (importance-based storage)
- Spatial decoder (attractor compression)
- Linked memory (cross-referencing)
"""

import pytest
import numpy as np
from grcm.echozero.integration.memory_router import MemoryRouter, AdaptiveMemoryRouter


class DummyHopfield:
    """Dummy Hopfield network for testing."""

    def __init__(self, dim=64):
        self.dim = dim
        self.W = np.random.randn(dim, dim)
        self.W = (self.W + self.W.T) / 2  # Make symmetric

    def step(self, state):
        new_state = self.W @ state
        return new_state / (np.linalg.norm(new_state) + 1e-8)


class TestMemoryRouter:
    """Test basic memory router functionality."""

    def test_initialization(self):
        """Test router initialization."""
        hopfield = DummyHopfield(dim=64)
        router = MemoryRouter(
            hopfield=hopfield,
            dim=64,
            semantic_capacity=100,
            decoder_rank=16,
            high_importance_threshold=1.0,
        )

        assert router.dim == 64
        assert router.high_importance_threshold == 1.0
        assert router.total_routed == 0
        assert router.hierarchical is not None
        assert router.semantic is not None
        assert router.decoder is not None
        assert router.linked is not None

    def test_route_low_importance_state(self):
        """Test routing state with low importance (no semantic storage)."""
        hopfield = DummyHopfield(dim=64)
        router = MemoryRouter(hopfield=hopfield, dim=64, high_importance_threshold=2.0)

        state = np.random.randn(64)
        result = router.route(state, torsion_score=0.5, attractor_idx=None)

        assert result["torsion"] == 0.5
        assert result["importance"] < 2.0
        assert result["pushed_to_semantic"] is False
        assert result["pushed_to_spatial"] is False

        # Should be in hierarchical replay
        assert router.total_routed == 1
        assert len(router.hierarchical.recent) == 1

    def test_route_high_importance_state(self):
        """Test routing high-importance state (with semantic storage)."""
        hopfield = DummyHopfield(dim=64)
        router = MemoryRouter(hopfield=hopfield, dim=64, high_importance_threshold=1.0)

        state = np.random.randn(64)
        result = router.route(state, torsion_score=5.0, attractor_idx=None)

        assert result["torsion"] == 5.0
        assert result["importance"] >= 1.0
        assert result["pushed_to_semantic"] is True
        assert result["pushed_to_spatial"] is False

        # Should be in both hierarchical and semantic
        assert len(router.hierarchical.recent) == 1
        assert len(router.semantic) == 1

    def test_route_with_attractor(self):
        """Test routing with attractor sync."""
        hopfield = DummyHopfield(dim=64)
        router = MemoryRouter(hopfield=hopfield, dim=64)

        state = np.random.randn(64)
        result = router.route(state, torsion_score=2.0, attractor_idx=3)

        assert result["attractor_idx"] == 3
        assert result["pushed_to_spatial"] is True
        assert router.attractor_sync_count == 1

        # Should have cross-reference
        assert router.linked.get_attractor_at_step(0) == 3

        # Should have spatial code
        assert len(router.decoder.codes) == 1

    def test_novelty_computation(self):
        """Test novelty score computation."""
        hopfield = DummyHopfield(dim=64)
        router = MemoryRouter(hopfield=hopfield, dim=64)

        # First state (no previous state, novelty = 0)
        state1 = np.ones(64)
        result1 = router.route(state1, torsion_score=1.0)
        assert result1["novelty"] == 0.0

        # Second state (different from first, novelty > 0)
        state2 = np.ones(64) * 5
        result2 = router.route(state2, torsion_score=1.0)
        assert result2["novelty"] > 0

    def test_importance_weighting(self):
        """Test importance score = 0.7 * torsion + 0.3 * novelty."""
        hopfield = DummyHopfield(dim=64)
        router = MemoryRouter(hopfield=hopfield, dim=64)

        # Route first state
        state1 = np.ones(64)
        router.route(state1, torsion_score=1.0)

        # Route second state with known difference
        state2 = np.ones(64) * 2
        result = router.route(state2, torsion_score=3.0)

        # Novelty = |state2 - state1| = |1*2 - 1*1| × sqrt(64) ≈ sqrt(64) = 8
        novelty = result["novelty"]
        expected_importance = 0.7 * 3.0 + 0.3 * novelty

        assert np.isclose(result["importance"], expected_importance, rtol=0.01)

    def test_get_high_importance_states(self):
        """Test retrieval of high-importance states."""
        hopfield = DummyHopfield(dim=64)
        router = MemoryRouter(hopfield=hopfield, dim=64, high_importance_threshold=1.0)

        # Route mix of high and low importance
        for i in range(10):
            torsion = 5.0 if i % 2 == 0 else 0.5
            router.route(np.random.randn(64), torsion_score=torsion)

        # Get high-importance (torsion >= 2.0)
        high_importance = router.get_high_importance_states(k=10, min_torsion=2.0)

        # Should get ~5 high-torsion states
        assert len(high_importance) >= 3
        for entry in high_importance:
            assert entry["torsion"] >= 2.0

    def test_stats_output(self):
        """Test comprehensive routing statistics."""
        hopfield = DummyHopfield(dim=64)
        router = MemoryRouter(hopfield=hopfield, dim=64, high_importance_threshold=1.0)

        # Route some states
        for i in range(20):
            torsion = 3.0 if i < 10 else 0.5
            attractor = i % 3 if i < 10 else None
            router.route(np.random.randn(64), torsion_score=torsion, attractor_idx=attractor)

        stats = router.stats()

        assert stats["total_routed"] == 20
        assert stats["high_importance_count"] > 0
        assert stats["attractor_sync_count"] == 10
        assert stats["high_importance_rate"] > 0
        assert stats["attractor_sync_rate"] == 0.5
        assert "hierarchical_stats" in stats
        assert "semantic_stats" in stats
        assert "linked_stats" in stats
        assert "decoder_stats" in stats


class TestAdaptiveMemoryRouter:
    """Test adaptive threshold adjustment."""

    def test_initialization(self):
        """Test adaptive router initialization."""
        hopfield = DummyHopfield(dim=64)
        router = AdaptiveMemoryRouter(
            hopfield=hopfield,
            dim=64,
            semantic_capacity=100,
            initial_threshold=1.0,
            target_fill_rate=0.8,
        )

        assert router.target_fill_rate == 0.8
        assert router.high_importance_threshold == 1.0

    def test_threshold_adaptation_upward(self):
        """Test threshold increases when buffer over-filled."""
        hopfield = DummyHopfield(dim=64)
        router = AdaptiveMemoryRouter(
            hopfield=hopfield,
            dim=64,
            semantic_capacity=10,
            initial_threshold=0.5,
            target_fill_rate=0.5,  # Target 50% fill
        )

        initial_threshold = router.high_importance_threshold

        # Route high-importance states to fill buffer beyond target
        for i in range(20):
            router.route(np.random.randn(64), torsion_score=5.0)

        # Threshold should increase (buffer over-filled)
        assert router.high_importance_threshold > initial_threshold

    def test_threshold_adaptation_downward(self):
        """Test threshold decreases when buffer under-filled."""
        hopfield = DummyHopfield(dim=64)
        router = AdaptiveMemoryRouter(
            hopfield=hopfield,
            dim=64,
            semantic_capacity=100,
            initial_threshold=10.0,  # Very high threshold
            target_fill_rate=0.8,
        )

        initial_threshold = router.high_importance_threshold

        # Route low-importance states (buffer under-filled)
        for i in range(20):
            router.route(np.random.randn(64), torsion_score=0.5)

        # Threshold should decrease (buffer under-filled)
        assert router.high_importance_threshold < initial_threshold

    def test_threshold_clamping(self):
        """Test threshold is clamped to [0.1, 10.0]."""
        hopfield = DummyHopfield(dim=64)
        router = AdaptiveMemoryRouter(
            hopfield=hopfield,
            dim=64,
            initial_threshold=0.01,  # Below min
        )

        # Route to trigger adaptation
        for i in range(20):
            router.route(np.random.randn(64), torsion_score=0.0)

        # Should be clamped above 0.1
        assert router.high_importance_threshold >= 0.1


def test_end_to_end_routing():
    """Test complete routing workflow."""
    hopfield = DummyHopfield(dim=64)
    router = MemoryRouter(hopfield=hopfield, dim=64, high_importance_threshold=1.0)

    # Simulate 100 steps with varying importance and attractor syncs
    for i in range(100):
        state = np.random.randn(64)
        torsion = np.random.uniform(0, 5)
        attractor = i % 10 if torsion > 2.0 else None

        router.route(state, torsion_score=torsion, attractor_idx=attractor)

    # Verify all systems have data
    assert len(router.hierarchical.recent) > 0
    assert router.total_routed == 100

    # Get stats
    stats = router.stats()
    assert stats["total_routed"] == 100
    assert stats["hierarchical_stats"]["total_states"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
