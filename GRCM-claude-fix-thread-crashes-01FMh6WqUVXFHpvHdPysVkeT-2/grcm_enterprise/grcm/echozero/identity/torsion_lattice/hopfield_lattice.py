"""
Hopfield Torsion Lattice - Vector Attractor Network for Identity Storage

Replaces XY-model physics with Hopfield-style energy descent:
- Multiple stable attractors (stored identity patterns)
- Hebbian learning from committed states
- Self-healing via energy minimization
- Topological binding to geometric spiral structure

Key Differences from XY-Model:
- XY: Single ground state (uniform) → information erasure
- Hopfield: Multiple attractors (patterns) → information persistence

Energy Function:
    E = -Σ_ij W_ij · (v_i · v_j) + Σ_i θ_i · ||v_i||²

Dynamics:
    v_i ← normalize(Σ_j W_ij v_j)

Weight Learning (Hebbian):
    W_ij += η · (p_i ⊗ p_j)  for each committed pattern p
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class HopfieldTorsionLattice:
    """
    3D Hopfield-style vector attractor network for identity storage.

    Uses Hebbian learning to create stable attractors for identity patterns.
    Each lattice node stores a 2D phase vector that evolves via energy descent.

    Capacity: ~0.14N patterns (N = size³ nodes)
    For 5³ = 125 nodes: ~17 stable patterns
    """

    def __init__(
        self,
        size: int = 5,
        learning_rate: float = 0.01,
        energy_threshold: float = 0.1,
        max_patterns: int = 20,
    ):
        """
        Initialize Hopfield torsion lattice.

        Args:
            size: Lattice dimension (creates size³ grid)
            learning_rate: Hebbian learning rate η
            energy_threshold: Minimum energy change for convergence
            max_patterns: Maximum number of stored patterns
        """
        self.size = size
        self.n_nodes = size ** 3
        self.learning_rate = learning_rate
        self.energy_threshold = energy_threshold
        self.max_patterns = max_patterns

        # State: 2D phase vectors at each lattice point
        # Shape: (size, size, size, 2) for (real, imag) components
        self.vectors = np.zeros((size, size, size, 2), dtype=np.float64)
        self._initialize_random()

        # Hebbian weight matrix: (n_nodes, n_nodes)
        # W[i,j] represents connection strength between nodes i and j
        self.W = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)

        # Stored patterns for tracking
        self.patterns = []
        self.pattern_energies = []

        # Statistics
        self.total_patterns_learned = 0
        self.learning_history = []

    def _initialize_random(self) -> None:
        """Initialize with small random vectors."""
        self.vectors = np.random.randn(self.size, self.size, self.size, 2) * 0.1
        # Normalize
        norms = np.linalg.norm(self.vectors, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        self.vectors = self.vectors / norms

    def _flatten_index(self, i: int, j: int, k: int) -> int:
        """Convert 3D lattice coordinates to flat index."""
        return i * self.size * self.size + j * self.size + k

    def _unflatten_index(self, idx: int) -> Tuple[int, int, int]:
        """Convert flat index to 3D lattice coordinates."""
        i = idx // (self.size * self.size)
        remainder = idx % (self.size * self.size)
        j = remainder // self.size
        k = remainder % self.size
        return i, j, k

    def _get_flat_state(self) -> np.ndarray:
        """Get flattened state vector (n_nodes, 2)."""
        return self.vectors.reshape(self.n_nodes, 2)

    def _set_from_flat_state(self, flat_state: np.ndarray) -> None:
        """Set lattice from flattened state vector."""
        self.vectors = flat_state.reshape(self.size, self.size, self.size, 2)

    def learn_pattern(self, pattern: np.ndarray) -> Dict:
        """
        Learn a new identity pattern using Hebbian rule.

        Hebbian Learning:
            W_ij += η · (p_i ⊗ p_j) / N

        This creates an attractor basin around the pattern.

        Args:
            pattern: 2D identity vector [real, imag]

        Returns:
            Dictionary with learning statistics
        """
        # Normalize input pattern
        pattern_norm = pattern / (np.linalg.norm(pattern) + 1e-8)

        # Write pattern to center region (3x3x3)
        center = self.size // 2
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    i, j, k = center + di, center + dj, center + dk
                    # Gaussian weight
                    dist_sq = di*di + dj*dj + dk*dk
                    weight = np.exp(-dist_sq / 2.0)
                    self.vectors[i, j, k] = pattern_norm * weight

        # Normalize all vectors
        norms = np.linalg.norm(self.vectors, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.vectors = self.vectors / norms

        # Get flattened pattern state
        flat_pattern = self._get_flat_state()

        # Hebbian learning: W += η · (p ⊗ p^T) / N
        # For vector patterns, we use outer product of full vectors
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:  # No self-connections
                    # Outer product contribution
                    contribution = np.dot(flat_pattern[i], flat_pattern[j])
                    self.W[i, j] += self.learning_rate * contribution / self.n_nodes

        # Symmetrize weight matrix
        self.W = (self.W + self.W.T) / 2.0

        # Store pattern
        if len(self.patterns) < self.max_patterns:
            self.patterns.append(pattern_norm.copy())
        else:
            # Replace oldest pattern
            self.patterns.pop(0)
            self.patterns.append(pattern_norm.copy())

        self.total_patterns_learned += 1

        # Compute pattern energy
        energy = self.compute_energy()

        return {
            'pattern_id': len(self.patterns) - 1,
            'total_learned': self.total_patterns_learned,
            'energy': energy,
            'capacity_used': len(self.patterns) / self.max_patterns
        }

    def step(self, steps: int = 1) -> None:
        """
        Perform Hopfield energy descent (replaces XY relaxation).

        Update rule:
            v_i(t+1) = normalize(Σ_j W_ij · v_j(t))

        This drives the system toward stored attractors.

        Args:
            steps: Number of descent steps
        """
        for _ in range(steps):
            flat_state = self._get_flat_state()
            new_state = np.zeros_like(flat_state)

            # Hopfield update: v_i ← Σ_j W_ij v_j
            for i in range(self.n_nodes):
                field = np.zeros(2)
                for j in range(self.n_nodes):
                    field += self.W[i, j] * flat_state[j]
                new_state[i] = field

            # Normalize vectors
            norms = np.linalg.norm(new_state, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            new_state = new_state / norms

            # Update state
            self._set_from_flat_state(new_state)

    def compute_energy(self) -> float:
        """
        Compute Hopfield energy.

        E = -Σ_ij W_ij · (v_i · v_j)

        Lower energy = closer to stored attractor.

        Returns:
            Current energy of the system
        """
        flat_state = self._get_flat_state()
        energy = 0.0

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):  # Avoid double-counting
                dot_product = np.dot(flat_state[i], flat_state[j])
                energy -= self.W[i, j] * dot_product

        return energy

    def write_vector(self, vec: np.ndarray, strength: float = 0.5) -> None:
        """
        Write identity vector to lattice center region.

        Uses strong write (0.5) since Hopfield will pull toward
        nearest attractor during relaxation.

        Args:
            vec: Identity vector [2D]
            strength: Write strength (higher than XY-model)
        """
        # Normalize
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)

        # Write to 3x3x3 center with Gaussian falloff
        center = self.size // 2
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    i, j, k = center + di, center + dj, center + dk

                    # Gaussian weight
                    dist_sq = di*di + dj*dj + dk*dk
                    weight = np.exp(-dist_sq / 2.0)
                    local_strength = strength * weight

                    # Blend with existing state
                    current = self.vectors[i, j, k]
                    new_vec = (1 - local_strength) * current + local_strength * vec_norm

                    # Normalize
                    norm = np.linalg.norm(new_vec)
                    if norm > 1e-8:
                        self.vectors[i, j, k] = new_vec / norm
                    else:
                        self.vectors[i, j, k] = vec_norm

    def read_vector(self) -> np.ndarray:
        """
        Read identity vector from lattice center region.

        Returns:
            Average 2D vector from center 3x3x3 region
        """
        center = self.size // 2
        vectors = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    i, j, k = center + di, center + dj, center + dk
                    vectors.append(self.vectors[i, j, k])

        # Vector average (not scalar average of components)
        vectors = np.array(vectors)
        avg_vec = np.mean(vectors, axis=0)

        # Normalize
        norm = np.linalg.norm(avg_vec)
        if norm > 1e-8:
            return avg_vec / norm
        else:
            return np.array([1.0, 0.0])

    def find_nearest_attractor(self, vec: np.ndarray) -> Tuple[int, float]:
        """
        Find which stored pattern is nearest to given vector.

        Args:
            vec: Query vector [2D]

        Returns:
            (pattern_id, distance) tuple
        """
        if len(self.patterns) == 0:
            return -1, float('inf')

        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)

        min_dist = float('inf')
        min_id = -1

        for i, pattern in enumerate(self.patterns):
            dist = np.linalg.norm(vec_norm - pattern)
            if dist < min_dist:
                min_dist = dist
                min_id = i

        return min_id, min_dist

    def get_magnetization(self) -> float:
        """
        Compute order parameter (for compatibility with XY-model tests).

        Returns:
            Average vector magnitude (coherence measure)
        """
        flat_state = self._get_flat_state()
        avg_vec = np.mean(flat_state, axis=0)
        return np.linalg.norm(avg_vec)

    def reset(self) -> None:
        """Reset lattice to random state."""
        self._initialize_random()

    def clear_patterns(self) -> None:
        """Clear all learned patterns and reset weights."""
        self.W = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        self.patterns = []
        self.pattern_energies = []
        self.total_patterns_learned = 0

    def __repr__(self) -> str:
        return (f"HopfieldTorsionLattice(size={self.size}, "
                f"patterns={len(self.patterns)}/{self.max_patterns}, "
                f"energy={self.compute_energy():.4f})")
