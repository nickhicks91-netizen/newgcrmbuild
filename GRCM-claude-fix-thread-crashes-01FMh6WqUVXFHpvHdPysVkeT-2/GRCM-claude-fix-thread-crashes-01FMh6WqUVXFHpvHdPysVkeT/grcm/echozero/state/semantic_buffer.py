"""
Semantic-Aware Replay Buffer with Importance-Based Eviction

Instead of pure FIFO eviction, this buffer:
- Evicts low-importance states (low torsion, low novelty)
- Preserves high-importance states (hallucinations, discontinuities)
- Uses weighted importance scoring
"""

import numpy as np
from typing import Dict, Any, Optional, List


class SemanticReplayBuffer:
    """
    Replay buffer that evicts based on semantic importance, not just age.

    Importance score = 0.7 * torsion + 0.3 * novelty

    High-torsion events (hallucinations, phase jumps) are preserved even if old.
    Low-torsion smooth states are evicted even if recent.
    """

    def __init__(self, capacity: int = 256, dim: int = 64):
        """
        Args:
            capacity: Maximum number of entries
            dim: State dimensionality
        """
        self.capacity = capacity
        self.dim = dim
        self.buffer: List[Dict[str, Any]] = []
        self.total_pushes = 0

    def _importance(self, entry: Dict[str, Any]) -> float:
        """
        Calculate importance score for an entry.

        Args:
            entry: Buffer entry with 'torsion' and 'novelty' fields

        Returns:
            Importance score (higher = more important)
        """
        torsion = entry.get("torsion", 0.0)
        novelty = entry.get("novelty", 0.0)

        # Weighted combination
        importance = 0.7 * torsion + 0.3 * novelty

        return float(importance)

    def _select_eviction_candidate(self) -> int:
        """
        Select buffer index to evict (lowest importance).

        Returns:
            Index of entry to evict
        """
        if len(self.buffer) == 0:
            return 0

        # Find entry with minimum importance
        min_idx = 0
        min_importance = self._importance(self.buffer[0])

        for i in range(1, len(self.buffer)):
            importance = self._importance(self.buffer[i])
            if importance < min_importance:
                min_importance = importance
                min_idx = i

        return min_idx

    def push(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Push entry to buffer, evicting lowest-importance entry if at capacity.

        Args:
            entry: Dictionary with keys:
                - 'state': np.ndarray of shape (dim,)
                - 'torsion': float (torsion score)
                - 'novelty': float (novelty score)
                - 'step': int (step number)
                - 'metadata': dict (optional)

        Returns:
            Evicted entry if one was removed, otherwise None
        """
        self.total_pushes += 1

        # Ensure entry has required fields
        if "state" not in entry:
            raise ValueError("Entry must have 'state' field")

        # Set defaults
        entry.setdefault("torsion", 0.0)
        entry.setdefault("novelty", 0.0)
        entry.setdefault("step", self.total_pushes - 1)
        entry.setdefault("metadata", {})

        # Copy state to prevent external mutation
        entry["state"] = np.copy(entry["state"]).astype(np.float32)

        evicted = None

        if len(self.buffer) < self.capacity:
            # Buffer not full, just append
            self.buffer.append(entry)
        else:
            # Buffer full, evict lowest-importance entry
            evict_idx = self._select_eviction_candidate()

            # Check if new entry is more important than weakest existing entry
            new_importance = self._importance(entry)
            evict_importance = self._importance(self.buffer[evict_idx])

            if new_importance > evict_importance:
                # Replace weakest entry
                evicted = self.buffer[evict_idx]
                self.buffer[evict_idx] = entry
            else:
                # New entry is less important, don't add it
                evicted = entry

        return evicted

    def reconstruct_exact(self, idx: int = -1) -> np.ndarray:
        """
        Retrieve state at index.

        Args:
            idx: Index into buffer (supports negative indexing)

        Returns:
            State vector
        """
        return np.copy(self.buffer[idx]["state"])

    def get_entry(self, idx: int = -1) -> Dict[str, Any]:
        """Get full entry (state + metadata) at index."""
        entry = self.buffer[idx]
        return {
            "state": np.copy(entry["state"]),
            "torsion": entry["torsion"],
            "novelty": entry["novelty"],
            "step": entry["step"],
            "metadata": entry.get("metadata", {}),
        }

    def get_high_importance_entries(self, k: int = 10, min_torsion: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get k highest-importance entries above threshold.

        Args:
            k: Number of entries to retrieve
            min_torsion: Minimum torsion threshold

        Returns:
            List of entries sorted by importance (descending)
        """
        # Filter by minimum torsion
        candidates = [e for e in self.buffer if e["torsion"] >= min_torsion]

        # Sort by importance
        candidates.sort(key=self._importance, reverse=True)

        # Return top-k
        return candidates[:k]

    def __len__(self) -> int:
        return len(self.buffer)

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all entries in buffer."""
        return [
            {
                "state": np.copy(e["state"]),
                "torsion": e["torsion"],
                "novelty": e["novelty"],
                "step": e["step"],
                "metadata": e.get("metadata", {}),
            }
            for e in self.buffer
        ]

    def stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "total_pushes": self.total_pushes,
                "avg_torsion": 0.0,
                "avg_novelty": 0.0,
                "avg_importance": 0.0,
                "max_importance": 0.0,
                "min_importance": 0.0,
            }

        torsions = [e["torsion"] for e in self.buffer]
        novelties = [e["novelty"] for e in self.buffer]
        importances = [self._importance(e) for e in self.buffer]

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "total_pushes": self.total_pushes,
            "avg_torsion": float(np.mean(torsions)),
            "avg_novelty": float(np.mean(novelties)),
            "avg_importance": float(np.mean(importances)),
            "max_importance": float(np.max(importances)),
            "min_importance": float(np.min(importances)),
        }
