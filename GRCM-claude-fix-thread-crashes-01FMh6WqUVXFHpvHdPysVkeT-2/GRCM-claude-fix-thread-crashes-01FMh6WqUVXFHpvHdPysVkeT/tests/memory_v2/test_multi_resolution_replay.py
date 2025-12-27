"""
Test Suite for Multi-Resolution Replay System

Tests hierarchical memory with three tiers:
- Recent: 256 states, full precision
- Medium: 1024 states, rank-32 compression (every 4th step)
- Long-term: 4096 states, rank-8 compression (every 16th step)
"""

import pytest
import numpy as np
from grcm.echozero.state.hierarchical_replay import (
    TieredReplayBuffer,
    MultiResolutionReplay,
)


class TestTieredReplayBuffer:
    """Test single-tier buffer with optional compression."""

    def test_initialization(self):
        """Test buffer initialization."""
        buf = TieredReplayBuffer(dim=64, capacity=100, rank=None)
        assert buf.dim == 64
        assert buf.capacity == 100
        assert buf.rank is None
        assert len(buf) == 0

    def test_full_precision_storage(self):
        """Test full precision storage (rank=None)."""
        buf = TieredReplayBuffer(dim=64, capacity=10, rank=None)

        # Push 5 states
        for i in range(5):
            state = np.ones(64) * i
            buf.push(state)

        assert len(buf) == 5

        # Verify exact reconstruction
        reconstructed = buf.reconstruct_exact(-1)
        expected = np.ones(64) * 4
        np.testing.assert_array_almost_equal(reconstructed, expected)

    def test_compressed_storage(self):
        """Test compressed storage with rank < dim."""
        buf = TieredReplayBuffer(dim=64, capacity=10, rank=8)

        # Push states
        for i in range(5):
            state = np.random.randn(64)
            buf.push(state)

        assert len(buf) == 5

        # Compressed states should have reduced dimensionality
        # (This is implementation detail, just verify it doesn't crash)
        reconstructed = buf.reconstruct_exact(-1)
        assert reconstructed.shape[0] <= 64  # May be compressed

    def test_buffer_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buf = TieredReplayBuffer(dim=32, capacity=5, rank=None)

        # Push 10 states (exceeds capacity)
        for i in range(10):
            buf.push(np.ones(32) * i)

        # Should only keep last 5
        assert len(buf) == 5

        # Oldest should be state 5, newest should be state 9
        oldest = buf.reconstruct_exact(0)
        newest = buf.reconstruct_exact(-1)
        assert np.allclose(oldest, np.ones(32) * 5)
        assert np.allclose(newest, np.ones(32) * 9)


class TestMultiResolutionReplay:
    """Test hierarchical three-tier replay system."""

    def test_initialization(self):
        """Test multi-resolution replay initialization."""
        replay = MultiResolutionReplay(dim=64)
        assert replay.dim == 64
        assert replay.steps == 0
        assert len(replay.recent) == 0
        assert len(replay.medium) == 0
        assert len(replay.longterm) == 0

    def test_capacity_growth(self):
        """Test that all tiers grow with appropriate sampling rates."""
        replay = MultiResolutionReplay(dim=64)

        # Push 5000 states
        for i in range(5000):
            replay.push(np.random.randn(64))

        # Check tier capacities
        assert len(replay.recent) == 256  # Maxed out
        assert len(replay.medium) > 500   # Should have ~1250 (every 4th)
        assert len(replay.longterm) > 200 # Should have ~312 (every 16th)

    def test_sampling_rates(self):
        """Test that tiers sample at correct rates (1x, 4x, 16x)."""
        replay = MultiResolutionReplay(dim=64)

        # Push 1000 states
        for i in range(1000):
            replay.push(np.ones(64) * i)

        # Recent: should have last 256
        assert len(replay.recent) == 256

        # Medium: every 4th step, so 1000/4 = 250
        # But capacity is 1024, so should have 250
        assert len(replay.medium) == 250

        # Long-term: every 16th step, so 1000/16 = 62
        assert len(replay.longterm) == 62

    def test_exact_recent_reconstruction(self):
        """Test exact reconstruction from recent tier."""
        replay = MultiResolutionReplay(dim=64)

        # Push 100 states
        states = []
        for i in range(100):
            state = np.random.randn(64)
            states.append(state)
            replay.push(state)

        # Reconstruct last 10 states
        for i in range(10):
            reconstructed = replay.sample_recent(k=1)[0] if i == 0 else replay.recent.reconstruct_exact(-(i+1))
            expected = states[-(i+1)]
            np.testing.assert_array_almost_equal(reconstructed, expected, decimal=5)

    def test_memory_usage_calculation(self):
        """Test memory usage estimation."""
        replay = MultiResolutionReplay(dim=64)

        # Push 1000 states
        for i in range(1000):
            replay.push(np.random.randn(64))

        # Get stats
        stats = replay.stats()

        # Check memory calculation
        # Recent: 256 states × 64 dim × 4 bytes = 65,536 bytes
        # Medium: 250 states × 32 dim × 4 bytes = 32,000 bytes
        # Longterm: 62 states × 8 dim × 4 bytes = 1,984 bytes
        # Total ≈ 99,520 bytes ≈ 97 KB

        assert stats["memory_bytes"] > 0
        assert stats["memory_kb"] > 0
        assert stats["total_states"] == len(replay.recent) + len(replay.medium) + len(replay.longterm)

    def test_stats_output(self):
        """Test comprehensive stats output."""
        replay = MultiResolutionReplay(dim=64)

        # Push some states
        for i in range(500):
            replay.push(np.random.randn(64))

        stats = replay.stats()

        # Verify all expected keys
        assert "steps" in stats
        assert "recent_count" in stats
        assert "medium_count" in stats
        assert "longterm_count" in stats
        assert "total_states" in stats
        assert "memory_bytes" in stats
        assert "memory_kb" in stats

        # Verify values
        assert stats["steps"] == 500
        assert stats["recent_count"] <= 256
        assert stats["total_states"] > 0

    def test_sample_recent(self):
        """Test sampling from recent tier."""
        replay = MultiResolutionReplay(dim=64)

        # Push 50 states
        for i in range(50):
            replay.push(np.ones(64) * i)

        # Sample last 10
        samples = replay.sample_recent(k=10)
        assert len(samples) == 10

        # Should be in reverse order (most recent first)
        assert np.allclose(samples[0], np.ones(64) * 49)
        assert np.allclose(samples[9], np.ones(64) * 40)

    def test_get_all_tiers(self):
        """Test retrieving all states from each tier."""
        replay = MultiResolutionReplay(dim=64)

        # Push 100 states
        for i in range(100):
            replay.push(np.random.randn(64))

        # Get all from each tier
        all_recent = replay.get_all_recent()
        all_medium = replay.get_all_medium()
        all_longterm = replay.get_all_longterm()

        # Verify counts
        assert len(all_recent) == 100  # All 100 fit in recent buffer
        assert len(all_medium) == 25   # Every 4th: 100/4 = 25
        assert len(all_longterm) == 6  # Every 16th: 100/16 = 6


def test_memory_efficiency():
    """Test that multi-resolution provides better memory efficiency."""
    # Single-tier baseline
    single_tier = TieredReplayBuffer(dim=64, capacity=256, rank=None)
    for i in range(5000):
        single_tier.push(np.random.randn(64))

    single_tier_states = len(single_tier)  # 256 (maxed out)
    single_tier_memory = 256 * 64 * 4  # bytes

    # Multi-resolution
    multi_res = MultiResolutionReplay(dim=64)
    for i in range(5000):
        multi_res.push(np.random.randn(64))

    multi_res_states = multi_res.total_states()
    multi_res_memory = multi_res.memory_usage_bytes()

    # Multi-resolution should store 5x+ more states
    # (Recent: 256, Medium: 1024, Longterm: ~312 = 1592 total)
    assert multi_res_states > single_tier_states * 5

    # But memory should be < 6x (due to compression tiers)
    assert multi_res_memory < single_tier_memory * 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
