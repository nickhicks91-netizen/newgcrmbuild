"""
Memory Router Integration Layer

Coordinates all memory subsystems:
- MultiResolutionReplay (temporal memory)
- SemanticReplayBuffer (importance-based eviction)
- HopfieldSpatialDecoder (attractor-guided compression)
- LinkedMemory (temporal ↔ attractor cross-reference)

Routes incoming states to appropriate memory tiers based on:
- Torsion score (importance)
- Novelty score (distance from previous states)
- Attractor activation (Hopfield sync events)
"""

import numpy as np
from typing import Optional, Dict, Any, List
from ..state.hierarchical_replay import MultiResolutionReplay
from ..state.semantic_buffer import SemanticReplayBuffer
from ..state.linked_memory import LinkedMemory
from ..memory.hopfield_decoder import HopfieldSpatialDecoder


class MemoryRouter:
    """
    Routes states to appropriate memory systems based on semantic importance.

    Three-tier routing:
    1. Hierarchical replay: All states (multi-resolution)
    2. Semantic buffer: High-importance states only
    3. Spatial decoder: Attractor states only (compressed)
    """

    def __init__(
        self,
        hopfield,
        dim: int = 64,
        semantic_capacity: int = 256,
        decoder_rank: int = 16,
        high_importance_threshold: float = 1.0,
    ):
        """
        Args:
            hopfield: Hopfield network instance
            dim: State dimensionality
            semantic_capacity: Capacity of semantic buffer
            decoder_rank: Compression rank for spatial decoder
            high_importance_threshold: Minimum importance score for semantic buffer
        """
        self.hopfield = hopfield
        self.dim = dim
        self.high_importance_threshold = high_importance_threshold

        # Initialize memory subsystems
        self.hierarchical = MultiResolutionReplay(dim=dim)
        self.semantic = SemanticReplayBuffer(capacity=semantic_capacity, dim=dim)
        self.linked = LinkedMemory(self.hierarchical)
        self.decoder = HopfieldSpatialDecoder(dim=dim, rank=decoder_rank)

        # Fit spatial decoder
        self.decoder.fit_projection(hopfield)

        # Statistics
        self.total_routed = 0
        self.high_importance_count = 0
        self.attractor_sync_count = 0
        self.prev_state = None

    def _compute_novelty(self, state: np.ndarray) -> float:
        """
        Compute novelty score (distance from previous state).

        Args:
            state: Current state

        Returns:
            Novelty score
        """
        if self.prev_state is None:
            return 0.0

        diff = np.linalg.norm(state - self.prev_state)
        return float(diff)

    def route(
        self,
        state: np.ndarray,
        torsion_score: float,
        attractor_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Route state to appropriate memory systems.

        Args:
            state: State vector (dim,)
            torsion_score: Torsion score (importance metric)
            attractor_idx: Hopfield attractor index (if synced)
            metadata: Optional metadata

        Returns:
            Routing summary dict
        """
        self.total_routed += 1

        # Compute novelty
        novelty = self._compute_novelty(state)

        # Tier 1: Push to hierarchical replay via linked memory (always)
        # Note: linked.push() calls replay.push() internally, so we don't call hierarchical.push() directly
        self.linked.push(state, attractor_idx=attractor_idx, metadata=metadata)

        # Tier 2: Push to semantic buffer if high importance
        importance = 0.7 * torsion_score + 0.3 * novelty
        pushed_to_semantic = False

        if importance >= self.high_importance_threshold:
            entry = {
                "state": state,
                "torsion": torsion_score,
                "novelty": novelty,
                "step": self.hierarchical.steps - 1,
                "metadata": metadata or {},
            }
            self.semantic.push(entry)
            pushed_to_semantic = True
            self.high_importance_count += 1

        # Tier 3: Store spatial code if attractor synced
        pushed_to_spatial = False
        if attractor_idx is not None:
            self.decoder.store_attractor(state)
            pushed_to_spatial = True
            self.attractor_sync_count += 1

        # Update previous state
        self.prev_state = np.copy(state)

        # Return routing summary
        return {
            "step": self.hierarchical.steps - 1,
            "torsion": torsion_score,
            "novelty": novelty,
            "importance": importance,
            "pushed_to_semantic": pushed_to_semantic,
            "pushed_to_spatial": pushed_to_spatial,
            "attractor_idx": attractor_idx,
        }

    def reconstruct_from_attractor(self, attractor_idx: int) -> np.ndarray:
        """
        Reconstruct state from spatial decoder.

        Args:
            attractor_idx: Index into stored attractor codes

        Returns:
            Reconstructed state
        """
        return self.decoder.reconstruct_attractor(attractor_idx)

    def get_attractor_examples(self, attractor_idx: int, k: int = 5) -> List[np.ndarray]:
        """
        Get exact examples of states that triggered this attractor.

        Args:
            attractor_idx: Attractor index
            k: Number of examples

        Returns:
            List of states
        """
        return self.linked.get_attractor_examples(attractor_idx, k=k)

    def get_high_importance_states(self, k: int = 10, min_torsion: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get k highest-importance states from semantic buffer.

        Args:
            k: Number of states
            min_torsion: Minimum torsion threshold

        Returns:
            List of state entries
        """
        return self.semantic.get_high_importance_entries(k=k, min_torsion=min_torsion)

    def stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        return {
            "total_routed": self.total_routed,
            "high_importance_count": self.high_importance_count,
            "high_importance_rate": self.high_importance_count / max(1, self.total_routed),
            "attractor_sync_count": self.attractor_sync_count,
            "attractor_sync_rate": self.attractor_sync_count / max(1, self.total_routed),
            "hierarchical_stats": self.hierarchical.stats(),
            "semantic_stats": self.semantic.stats(),
            "linked_stats": self.linked.stats(),
            "decoder_stats": self.decoder.stats(),
        }


class AdaptiveMemoryRouter(MemoryRouter):
    """
    Enhanced router with adaptive importance threshold.

    Automatically adjusts high_importance_threshold to maintain target
    semantic buffer fill rate (e.g., keep semantic buffer 80% full).
    """

    def __init__(
        self,
        hopfield,
        dim: int = 64,
        semantic_capacity: int = 256,
        decoder_rank: int = 16,
        initial_threshold: float = 1.0,
        target_fill_rate: float = 0.8,
        adaptation_rate: float = 0.01,
    ):
        """
        Args:
            hopfield: Hopfield network instance
            dim: State dimensionality
            semantic_capacity: Capacity of semantic buffer
            decoder_rank: Compression rank
            initial_threshold: Initial importance threshold
            target_fill_rate: Target semantic buffer fill rate (0-1)
            adaptation_rate: Threshold adaptation rate
        """
        super().__init__(
            hopfield=hopfield,
            dim=dim,
            semantic_capacity=semantic_capacity,
            decoder_rank=decoder_rank,
            high_importance_threshold=initial_threshold,
        )
        self.target_fill_rate = target_fill_rate
        self.adaptation_rate = adaptation_rate

    def route(
        self,
        state: np.ndarray,
        torsion_score: float,
        attractor_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route with adaptive threshold adjustment."""
        # Route normally
        result = super().route(state, torsion_score, attractor_idx, metadata)

        # Adapt threshold based on semantic buffer fill rate
        if self.total_routed % 10 == 0:  # Adapt every 10 steps
            current_fill_rate = len(self.semantic) / self.semantic.capacity

            if current_fill_rate < self.target_fill_rate:
                # Buffer under-filled, lower threshold
                self.high_importance_threshold *= (1 - self.adaptation_rate)
            else:
                # Buffer over-filled, raise threshold
                self.high_importance_threshold *= (1 + self.adaptation_rate)

            # Clamp threshold
            self.high_importance_threshold = max(0.1, min(10.0, self.high_importance_threshold))

        result["adaptive_threshold"] = self.high_importance_threshold
        return result
