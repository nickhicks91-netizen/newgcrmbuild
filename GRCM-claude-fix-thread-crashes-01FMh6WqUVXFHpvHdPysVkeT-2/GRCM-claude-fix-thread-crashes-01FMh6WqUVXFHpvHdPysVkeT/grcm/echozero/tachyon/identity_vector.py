"""
Identity Vector Manager

Sparse, update-safe identity representation that receives discrete
pattern indices from Hopfield classifier and performs soft updates.

Architecture:
- Stores identity as sparse vector (one entry per pattern type)
- Updates via soft reinforcement (not hard overwrites)
- Maintains temporal decay for episodic memory
- Safe for concurrent updates

This is the final output of the Tachyon pipeline:
  Torsion Lattice → Tachyon Detector → Hopfield Classifier → Identity Vector
"""

import numpy as np
from typing import Dict, List, Optional


class IdentityVector:
    """
    Sparse identity representation updated by event classifications.

    Each dimension corresponds to one event pattern type.
    Values represent "activation strength" of that pattern in current identity.

    Key features:
    - Soft updates: new_value = (1-α)·old + α·reinforcement
    - Temporal decay: values gradually fade without reinforcement
    - Bounded: values stay in [0, 1]
    - Sparse: most dimensions near zero
    """

    def __init__(
        self,
        num_patterns: int = 20,
        decay_rate: float = 0.01,
        update_strength: float = 0.05
    ):
        """
        Initialize identity vector.

        Args:
            num_patterns: Number of pattern dimensions (one per event type)
            decay_rate: Per-step decay rate (0 = no decay, 1 = immediate forget)
            update_strength: Reinforcement strength (0 = no learning, 1 = overwrite)
        """
        self.num_patterns = num_patterns
        self.decay_rate = decay_rate
        self.update_strength = update_strength

        # Identity state: sparse vector in [0, 1]^num_patterns
        self.state = np.zeros(num_patterns)

        # Statistics
        self.total_updates = 0
        self.update_history = []  # Recent pattern indices

    def reinforce(
        self,
        pattern_idx: int,
        strength: Optional[float] = None
    ):
        """
        Reinforce pattern dimension based on classified event.

        Soft update: state[idx] = (1-α)·state[idx] + α·strength

        Args:
            pattern_idx: Pattern index from Hopfield classifier
            strength: Optional override for update_strength
        """
        if pattern_idx < 0 or pattern_idx >= self.num_patterns:
            return

        alpha = strength if strength is not None else self.update_strength

        # Soft reinforcement
        self.state[pattern_idx] = (1 - alpha) * self.state[pattern_idx] + alpha * 1.0

        # Clip to [0, 1]
        self.state[pattern_idx] = np.clip(self.state[pattern_idx], 0, 1)

        # Track update
        self.total_updates += 1
        self.update_history.append(pattern_idx)

        # Keep history bounded
        if len(self.update_history) > 100:
            self.update_history.pop(0)

    def decay(self):
        """
        Apply temporal decay to all dimensions.

        Memories fade without reinforcement: state *= (1 - decay_rate)
        """
        self.state *= (1 - self.decay_rate)

        # Threshold very small values to exact zero (sparsity)
        self.state[self.state < 1e-6] = 0.0

    def get_dominant_patterns(self, top_k: int = 5) -> List[tuple]:
        """
        Get top-k most active pattern dimensions.

        Returns:
            List of (pattern_idx, activation_value) sorted by activation
        """
        indices = np.argsort(self.state)[::-1][:top_k]
        values = self.state[indices]

        result = [(int(idx), float(val)) for idx, val in zip(indices, values)]

        # Filter out zero activations
        result = [(idx, val) for idx, val in result if val > 1e-6]

        return result

    def get_state(self) -> np.ndarray:
        """Get full identity state vector."""
        return self.state.copy()

    def set_state(self, new_state: np.ndarray):
        """Set identity state (for loading saved state)."""
        if len(new_state) != self.num_patterns:
            raise ValueError(f"State dimension mismatch: {len(new_state)} != {self.num_patterns}")

        self.state = np.clip(new_state, 0, 1)

    def reset(self):
        """Reset identity to zero state."""
        self.state = np.zeros(self.num_patterns)
        self.total_updates = 0
        self.update_history = []

    def get_statistics(self) -> Dict:
        """Get identity statistics."""
        active_dims = np.sum(self.state > 1e-6)
        total_activation = np.sum(self.state)
        max_activation = np.max(self.state)

        dominant = self.get_dominant_patterns(top_k=3)

        return {
            'active_dimensions': int(active_dims),
            'total_activation': float(total_activation),
            'max_activation': float(max_activation),
            'sparsity': float(1 - active_dims / self.num_patterns),
            'total_updates': self.total_updates,
            'dominant_patterns': dominant
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"IdentityVector(active={stats['active_dimensions']}/{self.num_patterns}, "
                f"total_activation={stats['total_activation']:.3f})")
