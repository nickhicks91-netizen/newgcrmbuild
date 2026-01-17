"""
Test Suite for Linked Temporal ↔ Attractor Memory

Tests bidirectional mapping between:
- Temporal indices (replay buffer steps)
- Attractor indices (Hopfield attractor IDs)

Enables queries like:
- "Which states triggered attractor 5?"
- "Which attractor was active at step 100?"
"""

import pytest
import numpy as np
from grcm.echozero.state.hierarchical_replay import MultiResolutionReplay
from grcm.echozero.state.linked_memory import LinkedMemory


class TestLinkedMemory:
    """Test linked temporal ↔ attractor memory."""

    def test_initialization(self):
        """Test linked memory initialization."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        assert linked.replay is replay
        assert len(linked.attractor_to_replay) == 0
        assert len(linked.replay_to_attractor) == 0

    def test_push_without_attractor(self):
        """Test pushing state without attractor index."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        state = np.random.randn(64)
        linked.push(state, attractor_idx=None)

        assert replay.steps == 1
        assert len(linked.replay_to_attractor) == 0  # No attractor link

    def test_push_with_attractor(self):
        """Test pushing state with attractor index."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        state = np.random.randn(64)
        linked.push(state, attractor_idx=3)

        assert replay.steps == 1
        assert 0 in linked.replay_to_attractor
        assert linked.replay_to_attractor[0] == 3
        assert 3 in linked.attractor_to_replay
        assert 0 in linked.attractor_to_replay[3]

    def test_cross_reference_mapping(self):
        """Test bidirectional attractor ↔ replay mapping."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        # Push 10 states with alternating attractors
        for i in range(10):
            state = np.ones(64) * i
            attractor_idx = i % 2  # Alternates 0, 1, 0, 1, ...
            linked.push(state, attractor_idx=attractor_idx)

        # Check forward mapping (replay → attractor)
        assert linked.get_attractor_at_step(0) == 0
        assert linked.get_attractor_at_step(1) == 1
        assert linked.get_attractor_at_step(2) == 0
        assert linked.get_attractor_at_step(9) == 1

        # Check reverse mapping (attractor → replay)
        attractor_0_steps = linked.get_attractor_activation_history(0)
        attractor_1_steps = linked.get_attractor_activation_history(1)

        assert attractor_0_steps == [0, 2, 4, 6, 8]  # Even steps
        assert attractor_1_steps == [1, 3, 5, 7, 9]  # Odd steps

    def test_get_attractor_examples(self):
        """Test retrieving exact states for an attractor."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        # Push states with known patterns
        for i in range(10):
            state = np.ones(64) * i
            attractor_idx = 1 if i >= 5 else 0
            linked.push(state, attractor_idx=attractor_idx)

        # Get examples for attractor 1 (steps 5-9)
        examples = linked.get_attractor_examples(attractor_idx=1, k=3)

        # Should get last 3: steps 7, 8, 9
        assert len(examples) == 3
        np.testing.assert_array_almost_equal(examples[0], np.ones(64) * 7)
        np.testing.assert_array_almost_equal(examples[1], np.ones(64) * 8)
        np.testing.assert_array_almost_equal(examples[2], np.ones(64) * 9)

    def test_get_attractor_activation_counts(self):
        """Test counting attractor activations."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        # Push with varying attractor distributions
        # Attractor 0: 5 times
        # Attractor 1: 3 times
        # Attractor 2: 2 times
        for i in range(10):
            if i < 5:
                attractor_idx = 0
            elif i < 8:
                attractor_idx = 1
            else:
                attractor_idx = 2

            linked.push(np.random.randn(64), attractor_idx=attractor_idx)

        counts = linked.get_attractor_activation_counts()

        assert counts[0] == 5
        assert counts[1] == 3
        assert counts[2] == 2

    def test_find_attractor_transitions(self):
        """Test finding transitions between attractors."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        # Push with sequence: 0, 0, 1, 1, 2, 0
        attractor_sequence = [0, 0, 1, 1, 2, 0]

        for i, attractor_idx in enumerate(attractor_sequence):
            linked.push(np.random.randn(64), attractor_idx=attractor_idx)

        transitions = linked.find_attractor_transitions()

        # Expected transitions:
        # Step 2: 0 → 1
        # Step 4: 1 → 2
        # Step 5: 2 → 0

        assert len(transitions) == 3
        assert transitions[0] == (2, 0, 1)
        assert transitions[1] == (4, 1, 2)
        assert transitions[2] == (5, 2, 0)

    def test_get_dominant_attractor(self):
        """Test finding most frequently activated attractor."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        # Push with majority attractor 1
        # Attractor 0: 2 times
        # Attractor 1: 7 times
        # Attractor 2: 1 time
        attractor_sequence = [1, 1, 0, 1, 1, 2, 1, 0, 1, 1]

        for attractor_idx in attractor_sequence:
            linked.push(np.random.randn(64), attractor_idx=attractor_idx)

        dominant = linked.get_dominant_attractor()
        assert dominant == 1  # Most common

    def test_metadata_storage(self):
        """Test metadata storage and retrieval."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        # Push with metadata
        metadata = {"event": "hallucination", "confidence": 0.95}
        linked.push(np.random.randn(64), attractor_idx=0, metadata=metadata)

        # Retrieve metadata
        retrieved = linked.get_metadata_at_step(0)
        assert retrieved == metadata
        assert retrieved["event"] == "hallucination"
        assert retrieved["confidence"] == 0.95

    def test_stats_output(self):
        """Test comprehensive statistics."""
        replay = MultiResolutionReplay(dim=64)
        linked = LinkedMemory(replay)

        # Push some states
        for i in range(20):
            attractor_idx = i % 3  # Cycle through 0, 1, 2
            linked.push(np.random.randn(64), attractor_idx=attractor_idx)

        stats = linked.stats()

        assert stats["total_steps"] == 20
        assert stats["num_unique_attractors"] == 3
        assert stats["num_synced_steps"] == 20
        assert "activation_counts" in stats
        assert "replay_stats" in stats


def test_large_scale_cross_referencing():
    """Test cross-referencing at scale."""
    replay = MultiResolutionReplay(dim=64)
    linked = LinkedMemory(replay)

    # Push 1000 states with 10 attractors
    for i in range(1000):
        attractor_idx = i % 10
        linked.push(np.random.randn(64), attractor_idx=attractor_idx)

    # Each attractor should have 100 activations
    counts = linked.get_attractor_activation_counts()

    for attractor_idx in range(10):
        assert counts[attractor_idx] == 100

    # Check total unique attractors
    all_attractors = linked.get_all_attractor_indices()
    assert len(all_attractors) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
