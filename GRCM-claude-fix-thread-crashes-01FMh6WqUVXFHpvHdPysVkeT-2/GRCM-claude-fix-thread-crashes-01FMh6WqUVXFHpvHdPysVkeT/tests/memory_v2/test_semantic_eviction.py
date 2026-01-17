"""
Test Suite for Semantic-Aware Eviction

Tests importance-based memory eviction:
- High-importance states (high torsion, high novelty) are preserved
- Low-importance states are evicted even if recent
- Weighted importance scoring (70% torsion, 30% novelty)
"""

import pytest
import numpy as np
from grcm.echozero.state.semantic_buffer import SemanticReplayBuffer


class TestSemanticReplayBuffer:
    """Test semantic buffer with importance-based eviction."""

    def test_initialization(self):
        """Test buffer initialization."""
        buf = SemanticReplayBuffer(capacity=10, dim=64)
        assert buf.capacity == 10
        assert buf.dim == 64
        assert len(buf) == 0
        assert buf.total_pushes == 0

    def test_push_below_capacity(self):
        """Test pushing states when buffer is not full."""
        buf = SemanticReplayBuffer(capacity=5, dim=32)

        # Push 3 states
        for i in range(3):
            entry = {
                "state": np.ones(32) * i,
                "torsion": float(i),
                "novelty": 0.0,
            }
            evicted = buf.push(entry)
            assert evicted is None  # Nothing evicted yet

        assert len(buf) == 3

    def test_eviction_prefers_low_importance(self):
        """Test that eviction removes lowest-importance entries."""
        buf = SemanticReplayBuffer(capacity=4, dim=32)

        # Push entries with varying importance
        # Entry 0: torsion=0, importance=0
        # Entry 1: torsion=1, importance=0.7
        # Entry 2: torsion=2, importance=1.4
        # Entry 3: torsion=3, importance=2.1
        for i in range(4):
            entry = {
                "state": np.ones(32) * i,
                "torsion": float(i),
                "novelty": 0.0,
            }
            buf.push(entry)

        # Now push high-importance entry (should evict entry 0)
        high_importance = {
            "state": np.ones(32) * 10,
            "torsion": 5.0,
            "novelty": 0.0,
        }
        evicted = buf.push(high_importance)

        # Check that lowest-importance entry was evicted
        assert evicted is not None
        assert np.allclose(evicted["state"], np.ones(32) * 0)  # Entry 0 evicted

        # Highest torsion entries should remain
        torsions = [e["torsion"] for e in buf.buffer]
        assert 5.0 in torsions  # New high-importance entry
        assert 3.0 in torsions
        assert 2.0 in torsions
        assert 1.0 in torsions
        assert 0.0 not in torsions  # Lowest-importance evicted

    def test_importance_weighting(self):
        """Test that importance is weighted 70% torsion, 30% novelty."""
        buf = SemanticReplayBuffer(capacity=2, dim=16)

        # Entry 1: high torsion, low novelty
        # Importance = 0.7 * 10 + 0.3 * 0 = 7.0
        entry1 = {
            "state": np.ones(16),
            "torsion": 10.0,
            "novelty": 0.0,
        }

        # Entry 2: low torsion, high novelty
        # Importance = 0.7 * 0 + 0.3 * 10 = 3.0
        entry2 = {
            "state": np.ones(16) * 2,
            "torsion": 0.0,
            "novelty": 10.0,
        }

        buf.push(entry1)
        buf.push(entry2)

        # Entry 3: medium torsion, medium novelty
        # Importance = 0.7 * 5 + 0.3 * 5 = 5.0
        # Should evict entry2 (importance=3.0)
        entry3 = {
            "state": np.ones(16) * 3,
            "torsion": 5.0,
            "novelty": 5.0,
        }
        evicted = buf.push(entry3)

        assert evicted is not None
        assert evicted["torsion"] == 0.0  # Entry 2 evicted
        assert evicted["novelty"] == 10.0

    def test_get_high_importance_entries(self):
        """Test retrieval of high-importance entries."""
        buf = SemanticReplayBuffer(capacity=10, dim=32)

        # Push mix of high and low importance
        for i in range(10):
            entry = {
                "state": np.ones(32) * i,
                "torsion": float(i % 3),  # 0, 1, 2, 0, 1, 2, ...
                "novelty": 0.0,
            }
            buf.push(entry)

        # Get high-importance entries (torsion >= 2)
        high_importance = buf.get_high_importance_entries(k=5, min_torsion=2.0)

        # Should get entries with torsion=2 only
        assert len(high_importance) <= 5
        for entry in high_importance:
            assert entry["torsion"] >= 2.0

    def test_reconstruct_exact(self):
        """Test exact state reconstruction."""
        buf = SemanticReplayBuffer(capacity=5, dim=64)

        # Push known states
        for i in range(3):
            entry = {
                "state": np.ones(64) * i,
                "torsion": 1.0,
                "novelty": 0.0,
            }
            buf.push(entry)

        # Reconstruct
        state_0 = buf.reconstruct_exact(0)
        state_last = buf.reconstruct_exact(-1)

        np.testing.assert_array_almost_equal(state_0, np.ones(64) * 0)
        np.testing.assert_array_almost_equal(state_last, np.ones(64) * 2)

    def test_get_entry(self):
        """Test full entry retrieval (state + metadata)."""
        buf = SemanticReplayBuffer(capacity=5, dim=32)

        entry = {
            "state": np.random.randn(32),
            "torsion": 3.5,
            "novelty": 1.2,
            "step": 42,
            "metadata": {"event": "hallucination"},
        }
        buf.push(entry)

        # Retrieve full entry
        retrieved = buf.get_entry(-1)

        assert retrieved["torsion"] == 3.5
        assert retrieved["novelty"] == 1.2
        assert retrieved["step"] == 42
        assert retrieved["metadata"]["event"] == "hallucination"
        np.testing.assert_array_almost_equal(retrieved["state"], entry["state"])

    def test_stats_output(self):
        """Test statistics output."""
        buf = SemanticReplayBuffer(capacity=10, dim=64)

        # Empty buffer
        stats_empty = buf.stats()
        assert stats_empty["size"] == 0
        assert stats_empty["avg_torsion"] == 0.0

        # Push some entries
        for i in range(5):
            entry = {
                "state": np.random.randn(64),
                "torsion": float(i),
                "novelty": float(i) * 0.5,
            }
            buf.push(entry)

        stats = buf.stats()
        assert stats["size"] == 5
        assert stats["capacity"] == 10
        assert stats["total_pushes"] == 5
        assert stats["avg_torsion"] > 0
        assert stats["avg_importance"] > 0
        assert "max_importance" in stats
        assert "min_importance" in stats


def test_semantic_vs_fifo_preservation():
    """Test that semantic buffer preserves important events better than FIFO."""
    # Scenario: 10 states, capacity 5
    # States 0-4: low importance (torsion=0.1)
    # States 5-9: high importance (torsion=5.0)

    # FIFO would keep states 5-9 (last 5)
    # Semantic should also keep states 5-9 (high importance)

    buf = SemanticReplayBuffer(capacity=5, dim=32)

    # Push low-importance states
    for i in range(5):
        entry = {
            "state": np.ones(32) * i,
            "torsion": 0.1,
            "novelty": 0.0,
        }
        buf.push(entry)

    # Push high-importance states
    for i in range(5, 10):
        entry = {
            "state": np.ones(32) * i,
            "torsion": 5.0,
            "novelty": 0.0,
        }
        buf.push(entry)

    # All states should be high-importance
    torsions = [e["torsion"] for e in buf.buffer]
    assert all(t >= 4.0 for t in torsions)  # All high-importance

    # Now push low-importance state
    low_entry = {
        "state": np.ones(32) * 100,
        "torsion": 0.1,
        "novelty": 0.0,
    }
    evicted = buf.push(low_entry)

    # Low-importance entry should be immediately evicted or not added
    assert evicted is not None
    if evicted["torsion"] == 0.1:
        # New low-importance entry was rejected
        pass
    else:
        # One of the high-importance entries was evicted (shouldn't happen)
        assert False, "High-importance entry was evicted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
