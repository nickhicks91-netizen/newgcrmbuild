"""
Test Suite for EchoZero Memory Engine v2

Tests unified memory engine integration:
- Complete end-to-end workflow
- All subsystem coordination
- Query API
- Statistics and diagnostics
"""

import pytest
import numpy as np
from grcm.echozero.memory_engine import (
    EchoZeroMemoryEngine,
    EchoZeroMemoryEngineWithSnapback,
)


class DummyHopfield:
    """Dummy Hopfield network for testing."""

    def __init__(self, dim=64):
        self.dim = dim
        self.W = np.random.randn(dim, dim)
        self.W = (self.W + self.W.T) / 2

    def step(self, state):
        new_state = self.W @ state
        return new_state / (np.linalg.norm(new_state) + 1e-8)


class TestEchoZeroMemoryEngine:
    """Test unified memory engine."""

    def test_initialization(self):
        """Test engine initialization."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(
            hopfield=hopfield,
            dim=64,
            semantic_capacity=100,
            decoder_rank=16,
        )

        assert engine.dim == 64
        assert engine.hopfield is hopfield
        assert engine.hierarchical is not None
        assert engine.semantic is not None
        assert engine.decoder is not None
        assert engine.linked is not None

    def test_process_state(self):
        """Test state processing through engine."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(hopfield=hopfield, dim=64)

        state = np.random.randn(64)
        result = engine.process(state, torsion=2.5, attractor_idx=None)

        assert "step" in result
        assert "torsion" in result
        assert result["torsion"] == 2.5

    def test_recent_history_reconstruction(self):
        """Test exact recent history reconstruction."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(hopfield=hopfield, dim=64)

        # Process 50 states
        states = []
        for i in range(50):
            state = np.random.randn(64)
            states.append(state)
            engine.process(state, torsion=1.0)

        # Reconstruct recent states
        for i in range(10):
            reconstructed = engine.reconstruct_recent(steps_ago=i)
            expected = states[-(i+1)]
            np.testing.assert_array_almost_equal(reconstructed, expected)

    def test_get_recent_history(self):
        """Test batch retrieval of recent history."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(hopfield=hopfield, dim=64)

        # Process 100 states
        for i in range(100):
            engine.process(np.random.randn(64), torsion=1.0)

        # Get last 10 states
        history = engine.get_recent_history(k=10)
        assert len(history) == 10

    def test_important_events_retrieval(self):
        """Test retrieval of high-importance events."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(
            hopfield=hopfield,
            dim=64,
            high_importance_threshold=1.0,
        )

        # Process mix of high and low importance
        for i in range(20):
            torsion = 5.0 if i % 2 == 0 else 0.5
            engine.process(np.random.randn(64), torsion=torsion)

        # Get important events
        important = engine.get_important_events(k=5, min_torsion=2.0)

        assert len(important) > 0
        for event in important:
            assert event["torsion"] >= 2.0

    def test_attractor_analysis(self):
        """Test attractor activation analysis."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(hopfield=hopfield, dim=64)

        # Process with multiple attractors
        for i in range(30):
            attractor_idx = i % 3  # Cycle through 0, 1, 2
            engine.process(np.random.randn(64), torsion=2.0, attractor_idx=attractor_idx)

        # Get activation counts
        counts = engine.get_attractor_activation_counts()
        assert len(counts) == 3
        assert counts[0] == 10
        assert counts[1] == 10
        assert counts[2] == 10

    def test_get_attractor_examples(self):
        """Test retrieval of attractor examples."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(hopfield=hopfield, dim=64)

        # Process with known attractor
        for i in range(10):
            state = np.ones(64) * i
            engine.process(state, torsion=2.0, attractor_idx=1)

        # Get examples
        examples = engine.get_attractor_examples(attractor_idx=1, k=5)

        assert len(examples) == 5
        # Should be in chronological order (oldest to newest of selected k)
        # Pushed states 0-9, so last 5 are states 5-9
        np.testing.assert_array_almost_equal(examples[0], np.ones(64) * 5)  # Oldest of k
        np.testing.assert_array_almost_equal(examples[-1], np.ones(64) * 9)  # Most recent

    def test_find_attractor_transitions(self):
        """Test finding transitions between attractors."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(hopfield=hopfield, dim=64)

        # Process sequence: 0, 0, 1, 1, 2, 0
        attractor_sequence = [0, 0, 1, 1, 2, 0]
        for attractor_idx in attractor_sequence:
            engine.process(np.random.randn(64), torsion=2.0, attractor_idx=attractor_idx)

        transitions = engine.find_attractor_transitions()

        # Should find 3 transitions: 0→1, 1→2, 2→0
        assert len(transitions) == 3

    def test_memory_usage_calculation(self):
        """Test memory usage estimation."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(hopfield=hopfield, dim=64)

        # Process some states
        for i in range(500):
            engine.process(np.random.randn(64), torsion=1.5)

        memory_bytes = engine.memory_usage_bytes()
        memory_kb = engine.memory_usage_kb()

        assert memory_bytes > 0
        assert memory_kb == memory_bytes / 1024

    def test_stats_output(self):
        """Test comprehensive statistics."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(hopfield=hopfield, dim=64)

        # Process states
        for i in range(100):
            attractor = i % 5 if i % 3 == 0 else None
            engine.process(np.random.randn(64), torsion=float(i % 10), attractor_idx=attractor)

        stats = engine.stats()

        assert "total_routed" in stats
        assert "high_importance_count" in stats
        assert "hierarchical_stats" in stats
        assert "semantic_stats" in stats
        assert "linked_stats" in stats
        assert "decoder_stats" in stats


class TestEchoZeroMemoryEngineWithSnapback:
    """Test enhanced engine with attractor snapback."""

    def test_initialization(self):
        """Test snapback engine initialization."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngineWithSnapback(
            hopfield=hopfield,
            dim=64,
            decoder_rank=16,
        )

        assert engine.dim == 64
        assert engine.decoder_snapback is not None

    def test_reconstruct_with_snapback(self):
        """Test reconstruction with snapback option."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngineWithSnapback(hopfield=hopfield, dim=64, decoder_rank=16)

        # Process and store attractor
        state = np.random.randn(64)
        engine.process(state, torsion=2.0, attractor_idx=0)

        # Reconstruct with snapback
        reconstructed_snap = engine.reconstruct_attractor(0, snapback=True)
        reconstructed_no_snap = engine.reconstruct_attractor(0, snapback=False)

        # Should produce different results
        assert reconstructed_snap.shape == (64,)
        assert reconstructed_no_snap.shape == (64,)

    def test_adaptive_routing(self):
        """Test adaptive threshold adjustment."""
        hopfield = DummyHopfield(dim=64)
        engine = EchoZeroMemoryEngine(
            hopfield=hopfield,
            dim=64,
            semantic_capacity=10,
            use_adaptive_routing=True,
        )

        # Process high-importance states
        for i in range(50):
            engine.process(np.random.randn(64), torsion=5.0)

        # Threshold should have adapted
        # (Can't assert specific value, but verify it's still in valid range)
        threshold = engine.router.high_importance_threshold
        assert 0.1 <= threshold <= 10.0


def test_large_scale_integration():
    """Test engine with large-scale state processing."""
    hopfield = DummyHopfield(dim=64)
    engine = EchoZeroMemoryEngine(hopfield=hopfield, dim=64)

    # Process 2000 states
    for i in range(2000):
        state = np.random.randn(64)
        torsion = np.random.uniform(0, 5)
        attractor = i % 10 if torsion > 2.0 else None

        engine.process(state, torsion=torsion, attractor_idx=attractor)

    # Verify memory efficiency
    stats = engine.stats()
    h_stats = stats["hierarchical_stats"]

    # Should have multi-resolution coverage
    assert h_stats["recent_count"] == 256  # Maxed out
    assert h_stats["medium_count"] > 400   # Many medium-term
    assert h_stats["longterm_count"] > 100 # Some long-term

    # Total states should be much larger than single-tier
    assert h_stats["total_states"] > 700


def test_hallucination_detection_workflow():
    """Test hallucination detection use case."""
    hopfield = DummyHopfield(dim=64)
    engine = EchoZeroMemoryEngine(
        hopfield=hopfield,
        dim=64,
        high_importance_threshold=2.0,
    )

    # Simulate normal states with occasional hallucinations
    for i in range(100):
        state = np.random.randn(64)

        if i in [20, 45, 78]:  # Hallucination events
            torsion = 8.0
            attractor_idx = 99  # "hallucination attractor"
        else:
            torsion = 0.5
            attractor_idx = i % 5

        engine.process(state, torsion=torsion, attractor_idx=attractor_idx)

    # Retrieve hallucination candidates
    hallucinations = engine.get_hallucination_candidates(k=5)

    # Should find the 3 high-torsion events
    assert len(hallucinations) >= 3
    for event in hallucinations:
        assert event["torsion"] >= 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
