"""
Hopfield Torsion Lattice v5 - Center-Only with Pseudo-Inverse

CORRECT ARCHITECTURE: Matches pattern dimensionality to network dimensionality

Key insight: Patterns only differ in 3×3×3 center region (27 nodes).
Solution: Only compute 27×27 weight matrix for center, not full 125×125.

Architecture:
- Full lattice: 5×5×5 = 125 nodes (for spatial structure)
- Center region: 3×3×3 = 27 nodes (where patterns differ)
- Weight matrix: 27×27 complex (ONLY for center)
- Patterns: 27D complex vectors (center states)

This eliminates the 78% noise problem:
- Old approach: 125×125 weights, 78% encodes non-pattern noise
- New approach: 27×27 weights, 100% encodes actual patterns

Expected performance: ALL 5 TESTS PASSING
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class HopfieldTorsionLatticeCenter:
    """
    3D Hopfield network operating ONLY on center region with pseudo-inverse.

    Fixes architectural mismatch by matching pattern space to weight space.

    Architecture:
    - Full lattice: 5×5×5 grid for spatial embedding
    - Active region: 3×3×3 center (27 nodes with Hopfield weights)
    - Passive region: Outer shell (98 nodes with local dynamics)
    - Weights: 27×27 complex matrix via pseudo-inverse
    - Update: z_i ← normalize(Σ_j W[i,j] · z_j) for center only

    Capacity: ~0.14 × 27 ≈ 3-4 patterns (dense), or 10-15 with sparse codes
    """

    def __init__(
        self,
        size: int = 5,
        energy_threshold: float = 0.1,
        max_patterns: int = 20,
        orthogonality_threshold: float = 0.3,
    ):
        """
        Initialize center-only Hopfield lattice.

        Args:
            size: Full lattice dimension (creates size³ grid)
            energy_threshold: Convergence threshold
            max_patterns: Maximum stored patterns
            orthogonality_threshold: Warn if overlap > this value
        """
        self.size = size
        self.n_nodes = size ** 3

        # Center region: 3×3×3 = 27 nodes
        self.center_size = 3
        self.n_center = self.center_size ** 3  # 27
        self.center_start = (size - 3) // 2  # Index where center starts

        self.energy_threshold = energy_threshold
        self.max_patterns = max_patterns
        self.orthogonality_threshold = orthogonality_threshold

        # Full lattice state: Complex values at each point
        self.state = np.zeros((size, size, size), dtype=np.complex128)
        self._initialize_random()

        # Weight matrix: (27, 27) - ONLY for center region
        self.W_center = np.zeros((self.n_center, self.n_center), dtype=np.complex128)

        # Pattern storage
        self.patterns_2d = []  # Original 2D identity vectors
        self.pattern_centers = []  # 27D center states for each pattern

        # Statistics
        self.total_patterns_learned = 0
        self.last_recompute_energy = 0.0

    def _initialize_random(self) -> None:
        """Initialize with random unit complex numbers."""
        phases = np.random.uniform(0, 2*np.pi, size=(self.size, self.size, self.size))
        self.state = np.exp(1j * phases)

    def _vector_to_complex(self, vec: np.ndarray) -> complex:
        """Convert 2D vector to complex number: z = x + iy."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        return vec_norm[0] + 1j * vec_norm[1]

    def _complex_to_vector(self, z: complex) -> np.ndarray:
        """Convert complex number to 2D vector: [Re(z), Im(z)]."""
        z_norm = z / (abs(z) + 1e-8)
        return np.array([z_norm.real, z_norm.imag])

    def _get_center_region(self) -> np.ndarray:
        """Get 3×3×3 center region as (3,3,3) array."""
        c = self.center_start
        return self.state[c:c+3, c:c+3, c:c+3]

    def _set_center_region(self, center_array: np.ndarray) -> None:
        """Set 3×3×3 center region from (3,3,3) array."""
        c = self.center_start
        self.state[c:c+3, c:c+3, c:c+3] = center_array

    def _get_center_flat(self) -> np.ndarray:
        """Get flattened center region (27,)."""
        return self._get_center_region().flatten()

    def _set_center_flat(self, flat_center: np.ndarray) -> None:
        """Set center region from flattened (27,) array."""
        center_3d = flat_center.reshape(self.center_size, self.center_size, self.center_size)
        self._set_center_region(center_3d)

    def check_pattern_orthogonality(self, new_pattern: np.ndarray) -> Dict:
        """Check if new 2D pattern is sufficiently separated from existing ones."""
        new_norm = new_pattern / (np.linalg.norm(new_pattern) + 1e-8)

        overlaps = []
        max_overlap = 0.0
        max_overlap_idx = -1

        for i, existing in enumerate(self.patterns_2d):
            existing_norm = existing / (np.linalg.norm(existing) + 1e-8)
            overlap = abs(np.dot(new_norm, existing_norm))
            overlaps.append(overlap)

            if overlap > max_overlap:
                max_overlap = overlap
                max_overlap_idx = i

        return {
            'overlaps': overlaps,
            'max_overlap': max_overlap,
            'max_overlap_idx': max_overlap_idx,
            'orthogonal': max_overlap < self.orthogonality_threshold,
            'num_existing': len(self.patterns_2d)
        }

    def learn_patterns_batch(self, patterns_2d: List[np.ndarray]) -> Dict:
        """
        Learn multiple patterns using complex pseudo-inverse on CENTER ONLY.

        This is the key fix: compute 27×27 weights for center region where
        patterns actually differ, not 125×125 for full lattice.

        Args:
            patterns_2d: List of 2D identity vectors

        Returns:
            Learning statistics
        """
        if len(patterns_2d) == 0:
            return {'success': False, 'reason': 'no_patterns'}

        # Clear existing patterns
        self.patterns_2d = []
        self.pattern_centers = []

        # Normalize and store 2D patterns
        for p in patterns_2d[:self.max_patterns]:
            p_norm = p / (np.linalg.norm(p) + 1e-8)
            self.patterns_2d.append(p_norm)

        print(f"  Learning {len(self.patterns_2d)} patterns via center-only pseudo-inverse...")

        # For each 2D pattern, write to center and capture 27D center state
        # CRITICAL: Use same 2D→27D mapping here as in write_vector() for consistency!
        for i, pattern_2d in enumerate(self.patterns_2d):
            # Reset full lattice to ZERO (all point to +1+0j)
            self.state = np.ones((self.size, self.size, self.size), dtype=np.complex128)

            # Convert 2D vector to complex
            target_complex = self._vector_to_complex(pattern_2d)

            # Write using SAME method as write_vector() - Gaussian with full strength
            c = self.center_start
            for di in range(3):
                for dj in range(3):
                    for dk in range(3):
                        # Gaussian falloff from center
                        dist_sq = (di - 1)**2 + (dj - 1)**2 + (dk - 1)**2
                        weight = np.exp(-dist_sq / 2.0)  # Peak at center, falloff to neighbors

                        # Write pattern with Gaussian weight (full strength=1.0 for learning)
                        current = self.state[c+di, c+dj, c+dk]
                        new_val = (1 - weight) * current + weight * target_complex
                        self.state[c+di, c+dj, c+dk] = new_val / (abs(new_val) + 1e-8)

            # Capture ONLY the 27D center state (not full 125D lattice!)
            center_state = self._get_center_flat().copy()
            self.pattern_centers.append(center_state)

            if (i + 1) % 5 == 0:
                print(f"    Captured {i+1}/{len(self.patterns_2d)} center states (27D)")

        # Build pattern matrix P (27, n_patterns) - CENTER ONLY
        print(f"  Computing 27×27 pseudo-inverse weights...")
        pattern_matrix = np.zeros((self.n_center, len(self.pattern_centers)), dtype=np.complex128)

        for i, center_state in enumerate(self.pattern_centers):
            pattern_matrix[:, i] = center_state

        # Compute pseudo-inverse in complex domain - 27×27 only!
        try:
            P_pinv = np.linalg.pinv(pattern_matrix)
            self.W_center = pattern_matrix @ P_pinv  # (27, 27) complex matrix

            # KEEP diagonal for fixed-point property: W @ p = p
            # Standard Hopfield zeros diagonal, but pseudo-inverse NEEDS it for W @ P = P

            self.total_patterns_learned = len(self.patterns_2d)
            self.last_recompute_energy = self.compute_energy()

            print(f"  ✓ Center-only pseudo-inverse complete!")
            print(f"    Weight matrix: {self.W_center.shape} (27×27, not 125×125)")
            print(f"    Weight range: |W| ∈ [{abs(self.W_center).min():.4f}, {abs(self.W_center).max():.4f}]")
            print(f"    Energy: {self.last_recompute_energy:.4f}")

            return {
                'success': True,
                'num_patterns': len(self.patterns_2d),
                'weight_norm': np.linalg.norm(self.W_center),
                'energy': self.last_recompute_energy
            }

        except np.linalg.LinAlgError as e:
            print(f"  ✗ Pseudo-inverse failed: {e}")
            return {'success': False, 'reason': str(e)}

    def step(self, steps: int = 1) -> None:
        """
        Hopfield energy descent on CENTER REGION ONLY.

        Key fix: Only update the 27 center nodes with Hopfield dynamics.
        Rest of lattice stays passive or uses simple diffusion.
        """
        for _ in range(steps):
            # Get current center state (27D)
            center_flat = self._get_center_flat()

            # Hopfield update: field = W @ state (27D → 27D)
            center_new = self.W_center @ center_flat

            # Normalize to unit circle
            norms = np.abs(center_new)
            norms = np.maximum(norms, 1e-8)
            center_new = center_new / norms

            # Update only center region
            self._set_center_flat(center_new)

            # Optional: Let outer shell diffuse slowly (passive dynamics)
            # For now, leave outer shell unchanged

    def compute_energy(self) -> float:
        """Compute Hopfield energy for CENTER ONLY: E = -Re(Σ W[i,j]·conj(z_i)·z_j)"""
        center_flat = self._get_center_flat()

        # E = -0.5 * Re(z† W z) for 27D center
        energy = -0.5 * np.real(np.dot(center_flat.conj(), self.W_center @ center_flat))

        return energy

    def write_vector(self, vec: np.ndarray, strength: float = 0.5) -> None:
        """
        Write 2D identity vector with GAUSSIAN SPREAD (matching learning).

        During learning, patterns are written with Gaussian spread.
        During inference, we must use the SAME write method to create
        matching 27D states. We read from center voxel only.

        Args:
            vec: 2D vector to write
            strength: Write strength (0-1)
        """
        z = self._vector_to_complex(vec)

        # Write with Gaussian spread (SAME as learning)
        c = self.center_start
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    # Gaussian falloff from center
                    dist_sq = (di - 1)**2 + (dj - 1)**2 + (dk - 1)**2
                    weight = np.exp(-dist_sq / 2.0) * strength  # Peak at center

                    # Blend toward target
                    current = self.state[c+di, c+dj, c+dk]
                    new_val = (1 - weight) * current + weight * z
                    self.state[c+di, c+dj, c+dk] = new_val / (abs(new_val) + 1e-8)

    def read_vector(self) -> np.ndarray:
        """
        Read 2D identity vector from CENTER VOXEL ONLY.

        Returns:
            2D unit vector
        """
        # Read ONLY from center voxel
        c = self.center_start
        center_idx = (c + 1, c + 1, c + 1)
        center_complex = self.state[center_idx]

        # Normalize
        center_complex = center_complex / (abs(center_complex) + 1e-8)

        # Convert to 2D vector
        return self._complex_to_vector(center_complex)

    def find_nearest_attractor(self, vec: np.ndarray) -> Tuple[int, float]:
        """Find nearest stored pattern to given 2D vector."""
        if len(self.patterns_2d) == 0:
            return -1, float('inf')

        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)

        min_dist = float('inf')
        min_id = -1

        for i, pattern in enumerate(self.patterns_2d):
            dist = np.linalg.norm(vec_norm - pattern)
            if dist < min_dist:
                min_dist = dist
                min_id = i

        return min_id, min_dist

    def get_magnetization(self) -> float:
        """Compute center region order parameter |<z>|."""
        center_flat = self._get_center_flat()
        avg_z = np.mean(center_flat)
        return abs(avg_z)

    def reset(self) -> None:
        """Reset lattice to random state."""
        self._initialize_random()

    def __repr__(self) -> str:
        return (f"HopfieldTorsionLatticeCenter(size={self.size}, center={self.n_center}, "
                f"patterns={len(self.patterns_2d)}/{self.max_patterns}, "
                f"energy={self.compute_energy():.4f})")
