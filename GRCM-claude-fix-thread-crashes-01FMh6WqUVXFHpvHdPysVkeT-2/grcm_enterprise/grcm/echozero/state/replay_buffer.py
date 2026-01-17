"""
Replay Buffer for Exact State Reconstruction

Maintains a ring buffer of recent states with metadata for:
- Perfect reconstruction of recent history
- Causal sequence analysis
- Temporal coherence tracking
- Debug and interpretability
"""

import numpy as np
from collections import deque
from typing import Optional, List, Dict


class ReplayBuffer:
    """
    Reversible state buffer for exact reconstruction.

    Maintains last N input states plus metadata (torsion, gradients, etc.)
    This enables:
    - Exact reconstruction of recent states
    - Temporal sequence analysis
    - Causal chain debugging
    - Memory audit trails

    Unlike the spatial decoder (which gives approximate reconstruction),
    this buffer provides PERFECT reconstruction for recent history.

    Memory cost: O(capacity × dim) × 4 bytes
    For capacity=256, dim=64: ~64KB per session

    Example:
        >>> buffer = ReplayBuffer(dim=64, capacity=256)
        >>> buffer.push(state, torsion=1.5, phase_grad=0.3)
        >>> exact_state = buffer.reconstruct_exact(idx=-1)  # Last state
        >>> sequence = buffer.reconstruct_sequence(k=10)    # Last 10 states
    """

    def __init__(self, dim, capacity=256):
        """
        Initialize replay buffer.

        Args:
            dim: State vector dimensionality
            capacity: Maximum number of states to store (FIFO)
        """
        self.dim = dim
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        self.total_pushes = 0

    def push(self, state, torsion=None, phase_grad=None, metadata=None):
        """
        Add new state to buffer.

        Args:
            state: (dim,) state vector
            torsion: Optional torsion magnitude
            phase_grad: Optional phase gradient
            metadata: Optional dict of additional metadata
        """
        if state.shape[0] != self.dim:
            raise ValueError(f"State dim {state.shape[0]} != {self.dim}")

        entry = {
            "state": state.astype(np.float32).copy(),  # Ensure copy
            "torsion": float(torsion) if torsion is not None else 0.0,
            "phase_grad": float(phase_grad) if phase_grad is not None else 0.0,
            "step": self.total_pushes,
            "metadata": metadata or {},
        }

        self.buffer.append(entry)
        self.total_pushes += 1

    def last(self) -> Optional[Dict]:
        """
        Get last entry (most recent state).

        Returns:
            Dict with state, torsion, phase_grad, step, metadata
            Or None if buffer is empty
        """
        return self.buffer[-1] if self.buffer else None

    def reconstruct_exact(self, idx=-1) -> Optional[np.ndarray]:
        """
        Retrieve exact previous state.

        Args:
            idx: Index into buffer (-1 = most recent, -2 = second most recent, etc.)

        Returns:
            (dim,) state vector, or None if index out of range
        """
        try:
            return self.buffer[idx]["state"]
        except IndexError:
            return None

    def reconstruct_sequence(self, k=10) -> List[Dict]:
        """
        Retrieve last k states as a temporal sequence.

        Args:
            k: Number of recent states to retrieve

        Returns:
            List of state entries (most recent last)
        """
        return list(self.buffer)[-k:]

    def get_torsion_history(self, k=50) -> np.ndarray:
        """
        Get torsion magnitude history.

        Args:
            k: Number of recent steps

        Returns:
            Array of torsion values
        """
        recent = list(self.buffer)[-k:]
        return np.array([entry["torsion"] for entry in recent])

    def get_phase_gradient_history(self, k=50) -> np.ndarray:
        """
        Get phase gradient history.

        Args:
            k: Number of recent steps

        Returns:
            Array of phase gradients
        """
        recent = list(self.buffer)[-k:]
        return np.array([entry["phase_grad"] for entry in recent])

    def detect_discontinuities(self, threshold=5.0, window=20) -> List[int]:
        """
        Detect torsion discontinuities (spikes) in recent history.

        Args:
            threshold: Torsion magnitude threshold
            window: Number of recent steps to check

        Returns:
            List of step indices where discontinuities occurred
        """
        recent = list(self.buffer)[-window:]
        discontinuities = []

        for entry in recent:
            if entry["torsion"] > threshold:
                discontinuities.append(entry["step"])

        return discontinuities

    def compute_temporal_coherence(self, k=10) -> float:
        """
        Compute temporal coherence score.

        Measures smoothness of state transitions over last k steps.
        High coherence (near 1.0) = smooth transitions
        Low coherence (near 0.0) = chaotic transitions

        Args:
            k: Number of recent steps to analyze

        Returns:
            Coherence score in [0, 1]
        """
        recent = self.reconstruct_sequence(k=k)
        if len(recent) < 2:
            return 1.0

        # Compute state-to-state distances
        distances = []
        for i in range(1, len(recent)):
            s1 = recent[i-1]["state"]
            s2 = recent[i]["state"]
            dist = np.linalg.norm(s2 - s1)
            distances.append(dist)

        # Low mean distance = high coherence
        mean_dist = np.mean(distances)
        coherence = 1.0 / (1.0 + mean_dist)

        return float(coherence)

    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        if not self.buffer:
            return {
                'capacity': self.capacity,
                'size': 0,
                'total_pushes': self.total_pushes,
                'mean_torsion': 0.0,
                'max_torsion': 0.0,
            }

        torsion_vals = [e["torsion"] for e in self.buffer]

        return {
            'capacity': self.capacity,
            'size': len(self.buffer),
            'total_pushes': self.total_pushes,
            'mean_torsion': float(np.mean(torsion_vals)),
            'max_torsion': float(np.max(torsion_vals)),
            'min_torsion': float(np.min(torsion_vals)),
            'std_torsion': float(np.std(torsion_vals)),
        }

    def clear(self):
        """Clear all buffer contents."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return (f"ReplayBuffer(dim={self.dim}, capacity={self.capacity}, "
                f"size={len(self.buffer)}, pushes={self.total_pushes})")
