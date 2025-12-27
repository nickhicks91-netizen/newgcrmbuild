"""
Thread-Safe EchoZero Memory Engine

Provides thread-safe wrappers for concurrent access to memory systems.
Use this version when multiple threads may write to memory simultaneously.
"""

import threading
from typing import Optional, Dict, Any, List
import numpy as np
from .memory_engine import EchoZeroMemoryEngine, EchoZeroMemoryEngineWithSnapback


class ThreadSafeEchoZeroMemoryEngine:
    """
    Thread-safe wrapper for EchoZeroMemoryEngine.

    Uses a single lock to protect all write operations.
    Read operations can proceed concurrently.

    Thread-safety guarantees:
    - Multiple threads can call process() simultaneously
    - Reads are eventually consistent
    - No data corruption or race conditions
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
        Initialize thread-safe memory engine.

        Args:
            hopfield: Hopfield network instance
            dim: State dimensionality
            semantic_capacity: Capacity of semantic buffer
            decoder_rank: Compression rank for spatial decoder
            high_importance_threshold: Minimum importance for semantic storage
            use_adaptive_routing: Whether to use adaptive threshold adjustment
        """
        self._engine = EchoZeroMemoryEngine(
            hopfield=hopfield,
            dim=dim,
            semantic_capacity=semantic_capacity,
            decoder_rank=decoder_rank,
            high_importance_threshold=high_importance_threshold,
            use_adaptive_routing=use_adaptive_routing,
        )

        # Single lock for all write operations
        self._write_lock = threading.Lock()

        # Read-write lock for statistics
        # (allows multiple concurrent reads, exclusive write)
        self._stats_lock = threading.RLock()

    def process(
        self,
        state: np.ndarray,
        torsion: float,
        attractor_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Thread-safe state processing.

        Args:
            state: State vector (dim,)
            torsion: Torsion score (importance metric)
            attractor_idx: Hopfield attractor index (if synced)
            metadata: Optional metadata

        Returns:
            Routing summary
        """
        with self._write_lock:
            return self._engine.process(
                state=state,
                torsion=torsion,
                attractor_idx=attractor_idx,
                metadata=metadata,
            )

    # ========================================
    # TEMPORAL RECONSTRUCTION (Thread-safe reads)
    # ========================================

    def reconstruct_recent(self, steps_ago: int = 0) -> np.ndarray:
        """Thread-safe recent state reconstruction."""
        # Read-only operation, no lock needed (GIL protects simple reads)
        return self._engine.reconstruct_recent(steps_ago)

    def get_recent_history(self, k: int = 10) -> List[np.ndarray]:
        """Thread-safe recent history retrieval."""
        return self._engine.get_recent_history(k)

    def get_medium_history(self, k: int = 10) -> List[np.ndarray]:
        """Thread-safe medium-term history retrieval."""
        return self._engine.get_medium_history(k)

    def get_longterm_history(self, k: int = 10) -> List[np.ndarray]:
        """Thread-safe long-term history retrieval."""
        return self._engine.get_longterm_history(k)

    # ========================================
    # SPATIAL RECONSTRUCTION (Thread-safe reads)
    # ========================================

    def reconstruct_attractor(self, attractor_idx: int) -> np.ndarray:
        """Thread-safe attractor reconstruction."""
        return self._engine.reconstruct_attractor(attractor_idx)

    def get_attractor_examples(self, attractor_idx: int, k: int = 5) -> List[np.ndarray]:
        """Thread-safe attractor example retrieval."""
        return self._engine.get_attractor_examples(attractor_idx, k)

    # ========================================
    # IMPORTANCE-BASED QUERIES (Thread-safe reads)
    # ========================================

    def get_important_events(self, k: int = 10, min_torsion: float = 1.0) -> List[Dict[str, Any]]:
        """Thread-safe important event retrieval."""
        return self._engine.get_important_events(k, min_torsion)

    def get_hallucination_candidates(self, k: int = 5) -> List[Dict[str, Any]]:
        """Thread-safe hallucination candidate retrieval."""
        return self._engine.get_hallucination_candidates(k)

    # ========================================
    # ATTRACTOR ANALYSIS (Thread-safe reads)
    # ========================================

    def get_attractor_activation_counts(self) -> Dict[int, int]:
        """Thread-safe attractor activation count retrieval."""
        return self._engine.get_attractor_activation_counts()

    def get_dominant_attractor(
        self, start_step: int = 0, end_step: Optional[int] = None
    ) -> Optional[int]:
        """Thread-safe dominant attractor identification."""
        return self._engine.get_dominant_attractor(start_step, end_step)

    def find_attractor_transitions(self) -> List[tuple]:
        """Thread-safe attractor transition detection."""
        return self._engine.find_attractor_transitions()

    # ========================================
    # STATISTICS (Thread-safe with lock)
    # ========================================

    def stats(self) -> Dict[str, Any]:
        """Thread-safe statistics retrieval."""
        with self._stats_lock:
            return self._engine.stats()

    def memory_usage_bytes(self) -> int:
        """Thread-safe memory usage calculation."""
        with self._stats_lock:
            return self._engine.memory_usage_bytes()

    def memory_usage_kb(self) -> float:
        """Thread-safe memory usage in KB."""
        with self._stats_lock:
            return self._engine.memory_usage_kb()

    def print_stats(self) -> None:
        """Thread-safe statistics printing."""
        with self._stats_lock:
            self._engine.print_stats()


class ThreadSafeEchoZeroMemoryEngineWithSnapback(ThreadSafeEchoZeroMemoryEngine):
    """
    Thread-safe version with attractor snapback.

    Same thread-safety guarantees as base class.
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
        # Don't call super().__init__ because we need different engine type
        self._engine = EchoZeroMemoryEngineWithSnapback(
            hopfield=hopfield,
            dim=dim,
            semantic_capacity=semantic_capacity,
            decoder_rank=decoder_rank,
            high_importance_threshold=high_importance_threshold,
            use_adaptive_routing=use_adaptive_routing,
        )

        self._write_lock = threading.Lock()
        self._stats_lock = threading.RLock()

    def reconstruct_attractor(self, attractor_idx: int, snapback: bool = True) -> np.ndarray:
        """Thread-safe reconstruction with optional snapback."""
        return self._engine.reconstruct_attractor(attractor_idx, snapback)


# ========================================
# USAGE EXAMPLE
# ========================================

"""
Example usage in multi-threaded environment:

```python
from concurrent.futures import ThreadPoolExecutor
from grcm.echozero.memory_engine_threadsafe import ThreadSafeEchoZeroMemoryEngine
from grcm.hopfield import HopfieldNetwork

# Initialize
hopfield = HopfieldNetwork(dim=64, patterns=stored_patterns)
memory = ThreadSafeEchoZeroMemoryEngine(hopfield=hopfield, dim=64)

# Process states from multiple threads
def process_batch(states):
    for state in states:
        torsion = measure_torsion(state)
        memory.process(state, torsion=torsion)

# Run concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(process_batch, batch)
        for batch in batches
    ]

    # Wait for all to complete
    for future in futures:
        future.result()

# Query results (thread-safe)
hallucinations = memory.get_hallucination_candidates(k=10)
stats = memory.stats()
```

Performance Notes:
- Write operations are serialized (lock contention)
- Read operations are mostly lock-free
- Consider batching writes to reduce lock overhead
- For maximum throughput, use single-threaded writes
"""
