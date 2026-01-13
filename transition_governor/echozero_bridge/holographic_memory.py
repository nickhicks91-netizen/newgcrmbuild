"""
EchoZero Holographic Memory: Torsional Embedding Architecture

Based on "Physics of Meaning" framework. Implements:
- Complex torsional embeddings: z(t) = A · v_semantic · e^(iωt)
- Holographic State Matrix with Matrioshka decay
- Resonance-based retrieval
- Semantic gravity for hallucination correction

Phase 1: Proof of Concept - Memory enfolding and retrieval
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class TorsionalEmbedding:
    """
    Complex vector encoding semantic content with temporal phase.

    z(t) = A · v_semantic · e^(iωt)

    Components:
    - v_semantic: The content (token/concept embedding)
    - A (amplitude): Importance/attention weight (mass in semantic space)
    - ω (omega): Angular frequency (temporal rotation rate)
    - t: Timestamp
    """
    semantic_vector: np.ndarray  # Real vector (dimension N)
    amplitude: float  # Mass/importance (A >> 1 for critical data)
    omega: float  # Angular frequency (temporal signature)
    timestamp: float  # When this was encoded

    def to_complex_vector(self, query_time: Optional[float] = None) -> np.ndarray:
        """
        Convert to complex vector with phase rotation.

        Returns:
            Complex vector z(t) = A · v · e^(iωt)
        """
        t = query_time if query_time is not None else self.timestamp
        phase = np.exp(1j * self.omega * t)
        return self.amplitude * self.semantic_vector * phase


class HolographicMatrix:
    """
    Fixed-size holographic memory via superposition.

    Instead of linear append (KV-cache), uses:
    H_new = (H_old · δ) + z_input

    where δ (Matrioshka decay) creates nested time shells.
    """

    def __init__(
        self,
        dimension: int = 4096,
        decay_factor: float = 0.99,
        omega_base: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            dimension: Latent space size (N ≥ 1024 recommended)
            decay_factor: Matrioshka decay δ (e.g., 0.99)
            omega_base: Base angular frequency for temporal encoding
            seed: Random seed for reproducibility
        """
        if dimension < 1024:
            raise ValueError(f"Dimension {dimension} too low. Recommend N ≥ 1024 for SNR.")

        self.dimension = dimension
        self.decay_factor = decay_factor
        self.omega_base = omega_base

        # The holographic state matrix (complex-valued)
        self.H = np.zeros(dimension, dtype=np.complex128)

        # Metadata for retrieval
        self.enfolded_count = 0
        self.start_time = time.time()

        # Random number generator
        self.rng = np.random.RandomState(seed)

    def enfold(
        self,
        semantic_vector: np.ndarray,
        amplitude: float = 1.0,
        omega_scale: float = 1.0
    ) -> None:
        """
        Enfold new information into hologram via superposition.

        H_new = (H_old · δ) + z_input

        Args:
            semantic_vector: Content embedding (real vector)
            amplitude: Importance weight (mass)
            omega_scale: Multiplier for temporal frequency
        """
        if len(semantic_vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(semantic_vector)} != {self.dimension}")

        # Create torsional embedding
        current_time = time.time() - self.start_time
        omega = self.omega_base * omega_scale

        embedding = TorsionalEmbedding(
            semantic_vector=semantic_vector,
            amplitude=amplitude,
            omega=omega,
            timestamp=current_time
        )

        # Get complex vector at current time
        z = embedding.to_complex_vector()

        # Matrioshka decay + superposition
        self.H = (self.H * self.decay_factor) + z
        self.enfolded_count += 1

    def resonate(
        self,
        query_vector: np.ndarray,
        time_offset: float = 0.0,
        num_samples: int = 100
    ) -> Tuple[float, np.ndarray]:
        """
        Resonance-based retrieval via temporal phase matching.

        Recall = |⟨H, q · e^(-iωτ)⟩|

        Scans backwards through time to find constructive interference.

        Args:
            query_vector: What to search for
            time_offset: How far back to start search (seconds)
            num_samples: Number of temporal samples to try

        Returns:
            (resonance_strength, reconstructed_vector)
        """
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query dimension {len(query_vector)} != {self.dimension}")

        current_time = time.time() - self.start_time

        # Sample temporal phases
        time_samples = np.linspace(
            current_time - time_offset,
            current_time,
            num_samples
        )

        resonances = []
        reconstructed = []

        for t in time_samples:
            # Twist query backwards through time
            phase = np.exp(-1j * self.omega_base * t)
            q_twisted = query_vector * phase

            # Inner product with hologram
            resonance = np.abs(np.vdot(self.H, q_twisted))
            resonances.append(resonance)

            # If strong resonance, reconstruct
            if resonance > np.max(resonances) * 0.8:
                # Extract via conjugate multiplication
                reconstructed_complex = self.H * np.conj(q_twisted)
                reconstructed_real = np.real(reconstructed_complex)
                reconstructed.append((resonance, reconstructed_real))

        if not reconstructed:
            # No strong resonance, return zero vector
            return 0.0, np.zeros(self.dimension)

        # Return strongest resonance
        best_resonance, best_vector = max(reconstructed, key=lambda x: x[0])
        return best_resonance, best_vector

    def detect_attractors(self, threshold: float = 10.0) -> List[Tuple[int, float]]:
        """
        Detect high-mass regions (semantic attractors/potential hallucinations).

        Returns:
            List of (dimension_index, magnitude) for high-amplitude components
        """
        magnitudes = np.abs(self.H)
        mean_magnitude = np.mean(magnitudes)

        attractors = []
        for i, mag in enumerate(magnitudes):
            if mag > mean_magnitude * threshold:
                attractors.append((i, mag))

        return attractors

    def inject_counter_mass(
        self,
        semantic_vector: np.ndarray,
        amplitude: float,
        invert_phase: bool = True
    ) -> None:
        """
        Semantic Gravity correction via counter-mass injection.

        To fix a hallucination (high-mass error), inject equal/opposite mass
        to balance the metric and pull narrative toward truth.

        Args:
            semantic_vector: Truth vector
            amplitude: Counter-mass (should match or exceed error mass)
            invert_phase: If True, use opposite phase for cancellation
        """
        phase_multiplier = -1.0 if invert_phase else 1.0
        self.enfold(
            semantic_vector=semantic_vector,
            amplitude=amplitude,
            omega_scale=phase_multiplier
        )

    def get_context_retention(self, test_vectors: List[np.ndarray]) -> float:
        """
        Measure context retention: Can we retrieve old information?

        Args:
            test_vectors: Vectors that were previously enfolded

        Returns:
            Retention score (0-1, where 1 = perfect recall)
        """
        if not test_vectors:
            return 0.0

        retention_scores = []
        for vec in test_vectors:
            # Normalize input vector
            vec_norm = vec / (np.linalg.norm(vec) + 1e-10)

            # Query directly (recent memory)
            resonance_recent, _ = self.resonate(vec_norm, time_offset=1.0, num_samples=10)

            # Query with longer offset (older memory)
            resonance_old, _ = self.resonate(vec_norm, time_offset=100.0, num_samples=50)

            # Use max resonance as retention score
            retention = max(resonance_recent, resonance_old)

            # Normalize by expected magnitude (based on enfolding count)
            if self.enfolded_count > 0:
                expected_magnitude = np.linalg.norm(self.H) / np.sqrt(self.enfolded_count)
                retention = min(1.0, retention / (expected_magnitude + 1e-10))

            retention_scores.append(retention)

        return np.mean(retention_scores)

    def get_memory_stats(self) -> Dict:
        """Get hologram statistics."""
        return {
            'dimension': self.dimension,
            'enfolded_count': self.enfolded_count,
            'hologram_magnitude': np.linalg.norm(self.H),
            'mean_component': np.mean(np.abs(self.H)),
            'max_component': np.max(np.abs(self.H)),
            'decay_factor': self.decay_factor,
            'num_attractors': len(self.detect_attractors())
        }


class GovernorWithHolographicMemory:
    """
    Transition Governor + EchoZero Holographic Memory.

    Phase 1 integration: Governor uses hologram for context retrieval.
    """

    def __init__(
        self,
        governor,
        hologram_dimension: int = 4096,
        enable_memory: bool = True
    ):
        """
        Args:
            governor: TransitionGovernor instance
            hologram_dimension: Size of holographic memory
            enable_memory: If False, runs without hologram (baseline)
        """
        self.governor = governor
        self.enable_memory = enable_memory

        if enable_memory:
            self.hologram = HolographicMatrix(dimension=hologram_dimension)
        else:
            self.hologram = None

    def govern_with_memory(
        self,
        state,
        context_embedding: Optional[np.ndarray] = None,
        store_context: bool = True
    ):
        """
        Run governor with holographic context retrieval.

        Args:
            state: AIState
            context_embedding: Current context as embedding
            store_context: Whether to enfold this context

        Returns:
            (governor_output, memory_stats)
        """
        # Standard governance
        output = self.governor.govern(state)

        if not self.enable_memory or context_embedding is None:
            return output, {}

        # Store context if requested
        if store_context:
            # Use entropy as amplitude (high uncertainty = low mass)
            amplitude = 1.0 / (state.entropy + 0.1)
            self.hologram.enfold(context_embedding, amplitude=amplitude)

        # Retrieve related context via resonance
        resonance, recalled = self.hologram.resonate(context_embedding)

        # Check for attractors (potential hallucinations)
        attractors = self.hologram.detect_attractors()

        memory_stats = {
            'resonance_strength': resonance,
            'recalled_context_norm': np.linalg.norm(recalled),
            'num_attractors': len(attractors),
            'memory_magnitude': np.linalg.norm(self.hologram.H) if self.hologram else 0
        }

        return output, memory_stats
