"""
Linked Temporal ↔ Attractor Memory

Provides bidirectional mapping between:
- Temporal indices (step numbers in replay buffer)
- Attractor indices (Hopfield attractor IDs)

Enables queries like:
- "Show me all states that triggered attractor 5"
- "Which attractor was active at step 1000?"
- "Get the exact state history for this attractor"
"""

import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Any
from .hierarchical_replay import MultiResolutionReplay


class LinkedMemory:
    """
    Cross-reference system between temporal replay and attractor indices.

    Maintains two mappings:
    - attractor_to_replay: attractor_idx → [replay_step_1, replay_step_2, ...]
    - replay_to_attractor: replay_step → attractor_idx
    """

    def __init__(self, replay: MultiResolutionReplay):
        """
        Args:
            replay: MultiResolutionReplay instance to link with
        """
        self.replay = replay

        # Bidirectional mappings
        self.attractor_to_replay: Dict[int, List[int]] = defaultdict(list)
        self.replay_to_attractor: Dict[int, int] = {}

        # Metadata storage
        self.metadata: Dict[int, Dict[str, Any]] = {}

    def push(self, state: np.ndarray, attractor_idx: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Push state to replay buffer and create cross-reference.

        Args:
            state: State vector (dim,)
            attractor_idx: Hopfield attractor index (if synced)
            metadata: Optional metadata to store with this state
        """
        # Push to replay buffer
        self.replay.push(state)

        # Get current replay step
        replay_step = self.replay.steps - 1

        # Create cross-reference if attractor index provided
        if attractor_idx is not None:
            self.attractor_to_replay[attractor_idx].append(replay_step)
            self.replay_to_attractor[replay_step] = attractor_idx

        # Store metadata
        if metadata is not None:
            self.metadata[replay_step] = metadata

    def get_attractor_examples(self, attractor_idx: int, k: int = 5, tier: str = "recent") -> List[np.ndarray]:
        """
        Get k most recent exact states that triggered this attractor.

        Args:
            attractor_idx: Attractor index
            k: Number of examples to retrieve
            tier: Which replay tier to use ("recent", "medium", "longterm")

        Returns:
            List of k states (may be fewer if not enough states available)
        """
        if attractor_idx not in self.attractor_to_replay:
            return []

        # Get replay steps for this attractor
        replay_steps = self.attractor_to_replay[attractor_idx]

        # Get k most recent
        recent_steps = replay_steps[-k:]

        # Retrieve states from appropriate tier
        states = []
        current_step = self.replay.steps

        for step in recent_steps:
            # Calculate how many steps ago this was
            steps_ago = current_step - step - 1

            if tier == "recent":
                # Only available if within recent buffer capacity (256)
                if steps_ago < len(self.replay.recent):
                    idx = -(steps_ago + 1)
                    states.append(self.replay.recent.reconstruct_exact(idx))

            elif tier == "medium":
                # Check if available in medium tier
                # Medium stores every 4th step
                if step % 4 == 0:
                    medium_steps_ago = steps_ago // 4
                    if medium_steps_ago < len(self.replay.medium):
                        idx = -(medium_steps_ago + 1)
                        states.append(self.replay.medium.reconstruct_exact(idx))

            elif tier == "longterm":
                # Check if available in longterm tier
                # Longterm stores every 16th step
                if step % 16 == 0:
                    longterm_steps_ago = steps_ago // 16
                    if longterm_steps_ago < len(self.replay.longterm):
                        idx = -(longterm_steps_ago + 1)
                        states.append(self.replay.longterm.reconstruct_exact(idx))

        return states

    def get_attractor_at_step(self, step: int) -> Optional[int]:
        """
        Get attractor index that was active at given step.

        Args:
            step: Replay step number

        Returns:
            Attractor index, or None if no attractor was active
        """
        return self.replay_to_attractor.get(step)

    def get_metadata_at_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Get metadata stored at given step."""
        return self.metadata.get(step)

    def get_attractor_activation_history(self, attractor_idx: int) -> List[int]:
        """
        Get full history of steps when this attractor was active.

        Args:
            attractor_idx: Attractor index

        Returns:
            List of step numbers (sorted)
        """
        return sorted(self.attractor_to_replay[attractor_idx])

    def get_all_attractor_indices(self) -> List[int]:
        """Get list of all attractor indices that have been activated."""
        return sorted(self.attractor_to_replay.keys())

    def get_attractor_activation_counts(self) -> Dict[int, int]:
        """
        Get activation count for each attractor.

        Returns:
            Dict mapping attractor_idx → count
        """
        return {
            idx: len(steps)
            for idx, steps in self.attractor_to_replay.items()
        }

    def find_attractor_transitions(self) -> List[tuple]:
        """
        Find all transitions between attractors.

        Returns:
            List of (step, from_attractor, to_attractor) tuples
        """
        transitions = []
        prev_attractor = None

        for step in sorted(self.replay_to_attractor.keys()):
            curr_attractor = self.replay_to_attractor[step]

            if prev_attractor is not None and curr_attractor != prev_attractor:
                transitions.append((step, prev_attractor, curr_attractor))

            prev_attractor = curr_attractor

        return transitions

    def get_dominant_attractor(self, start_step: int = 0, end_step: Optional[int] = None) -> Optional[int]:
        """
        Get most frequently activated attractor in a time range.

        Args:
            start_step: Start of range (inclusive)
            end_step: End of range (exclusive), or None for all

        Returns:
            Attractor index with most activations, or None if no activations
        """
        if end_step is None:
            end_step = self.replay.steps

        # Count activations in range
        counts = defaultdict(int)

        for step, attractor_idx in self.replay_to_attractor.items():
            if start_step <= step < end_step:
                counts[attractor_idx] += 1

        if not counts:
            return None

        # Return most common
        return max(counts, key=counts.get)

    def stats(self) -> Dict[str, Any]:
        """Get linked memory statistics."""
        return {
            "total_steps": self.replay.steps,
            "num_unique_attractors": len(self.attractor_to_replay),
            "num_synced_steps": len(self.replay_to_attractor),
            "num_metadata_entries": len(self.metadata),
            "activation_counts": self.get_attractor_activation_counts(),
            "replay_stats": self.replay.stats(),
        }
