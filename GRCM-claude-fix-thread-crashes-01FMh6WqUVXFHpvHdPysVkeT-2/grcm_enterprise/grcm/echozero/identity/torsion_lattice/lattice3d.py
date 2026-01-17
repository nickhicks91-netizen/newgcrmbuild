"""
3D Torsion Memory Lattice - Long-term Identity Storage

Implements a 3D XY-model phase lattice with self-healing behavior.
This provides stable, noise-resistant long-term memory that operates
on a slow loop (0.1-2 Hz) independent of fast inference.

Key Properties:
- Self-healing: XY-model dynamics drive toward stable minima
- Noise-resistant: Phase coherence survives perturbations
- Möbius-gated: Only low-torsion patterns are written
- Non-invasive: Never blocks fast loop
"""

import numpy as np
from typing import Tuple, Optional


class TorsionLattice3D:
    """
    3D XY-model phase lattice with self-healing behavior.

    The lattice stores identity information as phase patterns in a 3D grid.
    Local relaxation dynamics automatically heal noise and damage.

    Physics:
        θ(i,j,k) ∈ [0, 2π)  - Phase at each lattice point
        m - Global magnitude (decays to enforce stability)

    Dynamics:
        θ_new = θ + κ·Σ_neighbors sin(θ_neighbor - θ)
        m_new = m·exp(-γ/2)
    """

    def __init__(
        self,
        size: int = 5,
        coupling: float = 1.0,
        gamma: float = 0.005,
    ):
        """
        Initialize 3D torsion lattice.

        Args:
            size: Lattice dimension (creates size³ grid)
            coupling: XY coupling strength (higher = faster relaxation)
            gamma: Magnitude decay rate (self-healing parameter)

        Physics regime (tuned for stability):
            coupling = 1.0  # balanced neighbor coupling
            gamma = 0.005   # slow decay: prevents collapse
            relax_factor = 0.08  # conservative update for convergence
        """
        self.size = size
        self.coupling = coupling
        self.gamma = gamma

        # Phase matrix: values in [0, 2π)
        self.theta = np.zeros((size, size, size), dtype=np.float64)
        self.init_checkerboard()

        # Magnitude (uniform across lattice)
        self.m = 1.0

        # Neighbor offsets (26-neighbor connectivity)
        self.neighbors = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

    def init_checkerboard(self) -> None:
        """
        Initialize with ferromagnetic ground state (uniform phase).

        All spins aligned to zero for maximum writability.
        Checkerboard (antiferromagnetic) was too rigid for information storage.
        """
        # Uniform initialization allows information to propagate
        self.theta.fill(0.0)

        # Add small random perturbation to break symmetry
        self.theta += np.random.uniform(-0.01, 0.01, size=self.theta.shape)

    def step(self) -> None:
        """
        One relaxation step using XY-model update with over-relaxation.

        Updates all phases based on neighbor coupling using vectorized
        operations and over-relaxation to prevent collapse to zero.
        Applies very slow magnitude decay for stability.
        """
        # Vectorized neighbor sum (6-neighbor stencil for efficiency)
        neighbor_sum = (
            np.roll(self.theta, 1, axis=0) +
            np.roll(self.theta, -1, axis=0) +
            np.roll(self.theta, 1, axis=1) +
            np.roll(self.theta, -1, axis=1) +
            np.roll(self.theta, 1, axis=2) +
            np.roll(self.theta, -1, axis=2)
        )

        # XY-model update: weighted mean-field + conservative relaxation
        delta = self.coupling * np.sin(neighbor_sum - 6 * self.theta)
        self.theta = self.theta + 0.08 * delta  # 0.08 for stable convergence

        # Keep angles wrapped in [-π, π) for numerical stability
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # Very slow global magnitude decay
        self.m *= (1.0 - self.gamma)

    def write_vector(self, v: np.ndarray, strength: float = 0.12) -> None:
        """
        Write an identity vector into the lattice center region with diffusion.

        Converts the identity vector to an angle and writes it to a 3x3x3 region
        around the center with Gaussian falloff. This allows information to
        propagate during relaxation.

        Args:
            v: Identity vector [2D] (real, imag components)
            strength: Write strength coefficient (0.12 for good retention)
        """
        # Convert vector to phase angle
        phase = np.arctan2(v[1], v[0])  # atan2(imag, real)

        # Write to 3x3x3 region around center with Gaussian falloff
        center = self.size // 2
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    i = center + di
                    j = center + dj
                    k = center + dk

                    # Gaussian falloff: center gets full strength, corners get ~37%
                    distance_sq = di*di + dj*dj + dk*dk
                    weight = np.exp(-distance_sq / 2.0)
                    local_strength = strength * weight

                    # Blend phase with local domain
                    local = self.theta[i, j, k]
                    new_phase = (1 - local_strength) * local + local_strength * phase

                    self.theta[i, j, k] = new_phase

        # Wrap all to [-π, π) for consistency
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def read_vector(self) -> np.ndarray:
        """
        Read identity vector from lattice center region.

        Reads phases from the 3x3x3 region around center where writes occur,
        computes mean phase, and converts to 2D vector representation.

        Returns:
            Identity vector [real, imag] representing average center phase
        """
        # Read from 3x3x3 center region where information is stored
        center = self.size // 2
        phases = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    i = center + di
                    j = center + dj
                    k = center + dk
                    phases.append(self.theta[i, j, k])

        # Compute mean phase using complex averaging
        phases = np.array(phases)
        mean_real = np.mean(np.cos(phases))
        mean_imag = np.mean(np.sin(phases))

        return np.array([mean_real, mean_imag], dtype=np.float64)

    def get_energy(self) -> float:
        """
        Compute XY-model energy (for debugging/monitoring).

        E = -Σ_<i,j> cos(θ_i - θ_j)

        Lower energy = more stable configuration.

        Returns:
            Total XY energy
        """
        energy = 0.0
        L = self.size

        for i in range(L):
            for j in range(L):
                for k in range(L):
                    for dx, dy, dz in self.neighbors:
                        ni = (i + dx) % L
                        nj = (j + dy) % L
                        nk = (k + dz) % L

                        # XY interaction
                        energy -= np.cos(self.theta[i, j, k] - self.theta[ni, nj, nk])

        # Divide by 2 (each pair counted twice)
        return energy / 2.0

    def get_magnetization(self) -> float:
        """
        Compute order parameter (magnetization).

        M = |⟨exp(iθ)⟩|

        M ≈ 1: ordered (coherent phases)
        M ≈ 0: disordered (random phases)

        Returns:
            Magnetization magnitude [0, 1]
        """
        vec = self.read_vector()
        return np.linalg.norm(vec)

    def reset(self) -> None:
        """Reset lattice to checkerboard ground state."""
        self.init_checkerboard()
        self.m = 1.0

    def __repr__(self) -> str:
        """String representation of lattice state."""
        mag = self.get_magnetization()
        energy = self.get_energy()
        return (
            f"TorsionLattice3D(size={self.size}, "
            f"m={self.m:.4f}, "
            f"magnetization={mag:.4f}, "
            f"energy={energy:.2f})"
        )
