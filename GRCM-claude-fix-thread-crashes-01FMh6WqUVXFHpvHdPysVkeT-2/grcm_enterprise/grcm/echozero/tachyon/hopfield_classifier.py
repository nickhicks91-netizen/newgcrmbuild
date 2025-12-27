"""
Hopfield Event Classifier

Simplified Hopfield network that classifies tachyon event signatures
into discrete pattern indices, NOT full spatial reconstruction.

Architecture:
- Input: 6D event signature vector
- Output: Pattern index (discrete label)
- Weight matrix: 6×6 (not 27×27)
- Purpose: Event classification, not spatial pattern storage

This avoids dimensional mismatch by operating directly on compact
event descriptors rather than distributed spatial representations.
"""

import numpy as np
from typing import List, Optional, Tuple


class HopfieldEventClassifier:
    """
    Discrete event classifier using Hopfield attractor dynamics.

    Stores prototype event signatures (6D vectors) and classifies
    new events by finding nearest attractor.

    Key difference from spatial Hopfield:
    - Operates on compact 6D signatures (not 27D spatial patterns)
    - Returns pattern INDEX (discrete), not reconstructed vector
    - No spatial encoding/decoding
    """

    def __init__(
        self,
        signature_dim: int = 6,
        max_patterns: int = 20,
        learning_rate: float = 0.1
    ):
        """
        Initialize Hopfield classifier.

        Args:
            signature_dim: Dimension of event signatures (6 by default)
            max_patterns: Maximum number of event types to learn
            learning_rate: Hebbian learning rate
        """
        self.signature_dim = signature_dim
        self.max_patterns = max_patterns
        self.learning_rate = learning_rate

        # Weight matrix: signature_dim × signature_dim
        self.W = np.zeros((signature_dim, signature_dim))

        # Stored prototype signatures
        self.prototypes = []  # List of 6D signature vectors
        self.pattern_labels = []  # Semantic labels for each pattern

    def learn_prototype(
        self,
        signature: np.ndarray,
        label: Optional[str] = None
    ) -> int:
        """
        Learn a new event prototype via Hebbian rule.

        Args:
            signature: 6D event signature vector
            label: Optional semantic label

        Returns:
            Pattern index assigned to this prototype
        """
        if len(self.prototypes) >= self.max_patterns:
            print(f"Warning: Max patterns ({self.max_patterns}) reached")
            return -1

        # Normalize signature
        sig_norm = signature / (np.linalg.norm(signature) + 1e-8)

        # Store prototype
        pattern_idx = len(self.prototypes)
        self.prototypes.append(sig_norm)
        self.pattern_labels.append(label or f"Pattern_{pattern_idx}")

        # Update weights via Hebbian rule: W += η·(s⊗s)
        self.W += self.learning_rate * np.outer(sig_norm, sig_norm)

        # Zero diagonal
        np.fill_diagonal(self.W, 0)

        print(f"  Learned prototype {pattern_idx}: {label}")

        return pattern_idx

    def learn_prototypes_batch(
        self,
        signatures: List[np.ndarray],
        labels: Optional[List[str]] = None
    ):
        """
        Learn multiple prototypes in batch.

        Uses pseudo-inverse for optimal weights.
        """
        if len(signatures) == 0:
            return

        # Limit to max_patterns
        signatures = signatures[:self.max_patterns]
        if labels:
            labels = labels[:self.max_patterns]
        else:
            labels = [f"Pattern_{i}" for i in range(len(signatures))]

        print(f"  Learning {len(signatures)} event prototypes...")

        # Normalize all signatures
        normalized = []
        for sig in signatures:
            sig_norm = sig / (np.linalg.norm(sig) + 1e-8)
            normalized.append(sig_norm)

        # Store prototypes
        self.prototypes = normalized
        self.pattern_labels = labels

        # Compute optimal weights via pseudo-inverse
        P = np.column_stack(normalized)  # (signature_dim, num_patterns)
        P_pinv = np.linalg.pinv(P)
        self.W = P @ P_pinv  # (signature_dim, signature_dim)

        # Zero diagonal
        np.fill_diagonal(self.W, 0)

        print(f"  ✓ Learned {len(self.prototypes)} prototypes")
        print(f"    Weight matrix: {self.W.shape}")

    def classify(
        self,
        signature: np.ndarray,
        relaxation_steps: int = 10
    ) -> Tuple[int, float]:
        """
        Classify event signature to nearest prototype index.

        Args:
            signature: 6D event signature
            relaxation_steps: Hopfield relaxation iterations

        Returns:
            (pattern_index, confidence)
        """
        if len(self.prototypes) == 0:
            return -1, 0.0

        # Normalize input
        state = signature / (np.linalg.norm(signature) + 1e-8)

        # Hopfield relaxation
        for _ in range(relaxation_steps):
            state = self.W @ state
            # Normalize
            norm = np.linalg.norm(state)
            if norm > 1e-8:
                state = state / norm

        # Find nearest prototype
        best_idx = -1
        best_similarity = -1.0

        for idx, prototype in enumerate(self.prototypes):
            similarity = np.dot(state, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx

        return best_idx, best_similarity

    def classify_direct(self, signature: np.ndarray) -> Tuple[int, float]:
        """
        Direct nearest-neighbor classification (no relaxation).

        Faster than Hopfield relaxation for well-separated prototypes.
        """
        if len(self.prototypes) == 0:
            return -1, 0.0

        sig_norm = signature / (np.linalg.norm(signature) + 1e-8)

        best_idx = -1
        best_similarity = -1.0

        for idx, prototype in enumerate(self.prototypes):
            similarity = np.dot(sig_norm, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx

        return best_idx, best_similarity

    def get_pattern_label(self, pattern_idx: int) -> str:
        """Get semantic label for pattern index."""
        if 0 <= pattern_idx < len(self.pattern_labels):
            return self.pattern_labels[pattern_idx]
        return f"Unknown_{pattern_idx}"

    def compute_energy(self, state: np.ndarray) -> float:
        """Compute Hopfield energy for given state."""
        return -0.5 * np.dot(state, self.W @ state)

    def reset(self):
        """Clear all learned prototypes."""
        self.W = np.zeros((self.signature_dim, self.signature_dim))
        self.prototypes = []
        self.pattern_labels = []

    def __repr__(self) -> str:
        return (f"HopfieldEventClassifier(prototypes={len(self.prototypes)}, "
                f"dim={self.signature_dim})")
