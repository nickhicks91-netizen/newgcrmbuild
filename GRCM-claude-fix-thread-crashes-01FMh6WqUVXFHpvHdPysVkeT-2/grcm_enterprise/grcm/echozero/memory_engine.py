"""
EchoZero Memory Engine v2 - Unified Memory System

Complete integration of all memory subsystems:
- Multi-resolution replay (17x more history)
- Semantic-aware eviction (preserve important events)
- Hopfield-guided compression (20-40% error vs 60-80%)
- Linked temporal ↔ attractor memory (cross-referencing)

Provides unified API for:
- State ingestion
- Exact temporal reconstruction
- Approximate spatial reconstruction
- Attractor-based retrieval
- Importance-based queries
"""

import numpy as np
from typing import Optional, Dict, Any, List
from .integration.memory_router import MemoryRouter, AdaptiveMemoryRouter
from .state.hierarchical_replay import MultiResolutionReplay
from .state.semantic_buffer import SemanticReplayBuffer
from .state.linked_memory import LinkedMemory
from .memory.hopfield_decoder import HopfieldSpatialDecoder, HopfieldSpatialDecoderWithSnapback


class EchoZeroMemoryEngine:
    """
    Unified memory engine for EchoZero.

    Provides complete memory management for AI inference pipelines:
    - 4,352 states in ~320KB (vs 256 states in 64KB)
    - Importance-based retention
    - Attractor-guided spatial compression
    - Temporal ↔ spatial cross-referencing
    """

    def __init__(
        self,
        hopfield,
        dim: int = 64,
        semantic_capacity: int = 256,
        decoder_rank: int = 16,
        high_importance_threshold: float = 1.0,
        use_adaptive_routing: bool = False,
    ):
        """
        Args:
            hopfield: Hopfield network instance
            dim: State dimensionality
            semantic_capacity: Capacity of semantic buffer
            decoder_rank: Compression rank for spatial decoder
            high_importance_threshold: Minimum importance for semantic storage
            use_adaptive_routing: Whether to use adaptive threshold adjustment
        """
        self.hopfield = hopfield
        self.dim = dim

        # Initialize router (with or without adaptive threshold)
        if use_adaptive_routing:
            self.router = AdaptiveMemoryRouter(
                hopfield=hopfield,
                dim=dim,
                semantic_capacity=semantic_capacity,
                decoder_rank=decoder_rank,
                initial_threshold=high_importance_threshold,
            )
        else:
            self.router = MemoryRouter(
                hopfield=hopfield,
                dim=dim,
                semantic_capacity=semantic_capacity,
                decoder_rank=decoder_rank,
                high_importance_threshold=high_importance_threshold,
            )

        # Expose subsystems for direct access
        self.hierarchical = self.router.hierarchical
        self.semantic = self.router.semantic
        self.linked = self.router.linked
        self.decoder = self.router.decoder

    def process(
        self,
        state: np.ndarray,
        torsion: float,
        attractor_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process incoming state through all memory systems.

        Args:
            state: State vector (dim,)
            torsion: Torsion score (importance metric)
            attractor_idx: Hopfield attractor index (if synced)
            metadata: Optional metadata

        Returns:
            Routing summary
        """
        return self.router.route(
            state=state,
            torsion_score=torsion,
            attractor_idx=attractor_idx,
            metadata=metadata,
        )

    # ========================================
    # TEMPORAL RECONSTRUCTION
    # ========================================

    def reconstruct_recent(self, steps_ago: int = 0) -> np.ndarray:
        """
        Reconstruct exact state from recent history.

        Args:
            steps_ago: How many steps back (0 = most recent)

        Returns:
            Exact state vector
        """
        return self.hierarchical.recent.reconstruct_exact(-(steps_ago + 1))

    def get_recent_history(self, k: int = 10) -> List[np.ndarray]:
        """Get k most recent states (exact)."""
        return self.hierarchical.sample_recent(k=k)

    def get_medium_history(self, k: int = 10) -> List[np.ndarray]:
        """Get k most recent medium-term states (compressed)."""
        return self.hierarchical.sample_medium(k=k)

    def get_longterm_history(self, k: int = 10) -> List[np.ndarray]:
        """Get k most recent long-term states (heavily compressed)."""
        return self.hierarchical.sample_longterm(k=k)

    # ========================================
    # SPATIAL RECONSTRUCTION
    # ========================================

    def reconstruct_attractor(self, attractor_idx: int) -> np.ndarray:
        """
        Reconstruct state from spatial decoder.

        Args:
            attractor_idx: Index into stored attractor codes

        Returns:
            Approximate reconstructed state
        """
        return self.router.reconstruct_from_attractor(attractor_idx)

    def get_attractor_examples(self, attractor_idx: int, k: int = 5) -> List[np.ndarray]:
        """
        Get exact examples of states that triggered this attractor.

        Args:
            attractor_idx: Attractor index
            k: Number of examples

        Returns:
            List of exact states
        """
        return self.router.get_attractor_examples(attractor_idx, k=k)

    # ========================================
    # IMPORTANCE-BASED QUERIES
    # ========================================

    def get_important_events(self, k: int = 10, min_torsion: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get k most important events (high torsion/novelty).

        Args:
            k: Number of events
            min_torsion: Minimum torsion threshold

        Returns:
            List of event entries
        """
        return self.router.get_high_importance_states(k=k, min_torsion=min_torsion)

    def get_hallucination_candidates(self, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top-k hallucination candidates (highest torsion).

        Returns:
            List of high-torsion events
        """
        return self.get_important_events(k=k, min_torsion=2.0)

    # ========================================
    # ATTRACTOR ANALYSIS
    # ========================================

    def get_attractor_activation_counts(self) -> Dict[int, int]:
        """Get activation count for each attractor."""
        return self.linked.get_attractor_activation_counts()

    def get_dominant_attractor(self, start_step: int = 0, end_step: Optional[int] = None) -> Optional[int]:
        """Get most frequently activated attractor in time range."""
        return self.linked.get_dominant_attractor(start_step=start_step, end_step=end_step)

    def find_attractor_transitions(self) -> List[tuple]:
        """Find all transitions between attractors."""
        return self.linked.find_attractor_transitions()

    # ========================================
    # STATISTICS AND DIAGNOSTICS
    # ========================================

    def stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return self.router.stats()

    def memory_usage_bytes(self) -> int:
        """Estimate total memory usage in bytes."""
        stats = self.stats()
        hierarchical_bytes = stats["hierarchical_stats"]["memory_bytes"]
        decoder_bytes = stats["decoder_stats"]["memory_bytes"]

        # Approximate semantic buffer size
        semantic_bytes = len(self.semantic) * self.dim * 4  # float32

        return hierarchical_bytes + decoder_bytes + semantic_bytes

    def memory_usage_kb(self) -> float:
        """Memory usage in KB."""
        return self.memory_usage_bytes() / 1024

    def print_stats(self) -> None:
        """Print formatted statistics."""
        stats = self.stats()

        print("=" * 60)
        print("EchoZero Memory Engine v2 - Statistics")
        print("=" * 60)

        print(f"\nRouting:")
        print(f"  Total states routed: {stats['total_routed']}")
        print(f"  High-importance rate: {stats['high_importance_rate']:.2%}")
        print(f"  Attractor sync rate: {stats['attractor_sync_rate']:.2%}")

        h_stats = stats["hierarchical_stats"]
        print(f"\nHierarchical Memory:")
        print(f"  Recent tier: {h_stats['recent_count']}/256 states")
        print(f"  Medium tier: {h_stats['medium_count']}/1024 states")
        print(f"  Longterm tier: {h_stats['longterm_count']}/4096 states")
        print(f"  Total states: {h_stats['total_states']}")
        print(f"  Memory usage: {h_stats['memory_kb']:.1f} KB")

        s_stats = stats["semantic_stats"]
        print(f"\nSemantic Buffer:")
        print(f"  Size: {s_stats['size']}/{s_stats['capacity']}")
        print(f"  Avg torsion: {s_stats['avg_torsion']:.3f}")
        print(f"  Avg importance: {s_stats['avg_importance']:.3f}")

        l_stats = stats["linked_stats"]
        print(f"\nLinked Memory:")
        print(f"  Unique attractors: {l_stats['num_unique_attractors']}")
        print(f"  Synced steps: {l_stats['num_synced_steps']}")

        d_stats = stats["decoder_stats"]
        print(f"\nSpatial Decoder:")
        print(f"  Compression: {d_stats['compression_ratio']:.1f}x")
        print(f"  Stored codes: {d_stats['num_codes']}")
        print(f"  Memory: {d_stats['memory_kb']:.1f} KB")

        print(f"\nTotal Memory: {self.memory_usage_kb():.1f} KB")
        print("=" * 60)


class EchoZeroMemoryEngineWithSnapback(EchoZeroMemoryEngine):
    """
    Enhanced memory engine with attractor snapback for improved reconstruction.

    Uses HopfieldSpatialDecoderWithSnapback to project reconstructions
    onto nearest Hopfield attractor, reducing reconstruction error.
    """

    def __init__(
        self,
        hopfield,
        dim: int = 64,
        semantic_capacity: int = 256,
        decoder_rank: int = 16,
        high_importance_threshold: float = 1.0,
        use_adaptive_routing: bool = False,
    ):
        # Initialize parent
        super().__init__(
            hopfield=hopfield,
            dim=dim,
            semantic_capacity=semantic_capacity,
            decoder_rank=decoder_rank,
            high_importance_threshold=high_importance_threshold,
            use_adaptive_routing=use_adaptive_routing,
        )

        # Replace decoder with snapback version
        self.decoder_snapback = HopfieldSpatialDecoderWithSnapback(
            dim=dim,
            rank=decoder_rank,
            hopfield=hopfield,
        )
        self.decoder_snapback.fit_projection(hopfield)

        # Update router's decoder
        self.router.decoder = self.decoder_snapback
        self.decoder = self.decoder_snapback

    def reconstruct_attractor(self, attractor_idx: int, snapback: bool = True) -> np.ndarray:
        """
        Reconstruct with optional snapback.

        Args:
            attractor_idx: Index into stored codes
            snapback: Whether to project onto nearest attractor

        Returns:
            Reconstructed state
        """
        return self.decoder_snapback.reconstruct_attractor(attractor_idx, snapback=snapback)
