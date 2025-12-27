"""
Identity Readout - Extract Long-term Memory

Utilities for reading identity vectors from the 3D torsion lattice.
Provides safe, non-blocking access to lattice state.
"""

import numpy as np
from typing import Optional, Dict, Any
from .lattice3d import TorsionLattice3D


def read_identity(lattice: TorsionLattice3D) -> np.ndarray:
    """
    Reads lattice identity vector as 2D phase average.

    This is a simple wrapper around lattice.read_vector()
    for consistency with the module API.

    Args:
        lattice: TorsionLattice3D instance

    Returns:
        Identity vector [real, imag] representing average phase
    """
    return lattice.read_vector()


def read_identity_with_stats(lattice: TorsionLattice3D) -> Dict[str, Any]:
    """
    Read identity vector with additional statistics.

    Provides comprehensive readout including coherence metrics.

    Args:
        lattice: TorsionLattice3D instance

    Returns:
        Dictionary with:
            - identity: Identity vector [2D]
            - magnetization: Order parameter [0, 1]
            - energy: XY-model energy
            - magnitude: Global magnitude factor
    """
    identity = lattice.read_vector()
    magnetization = lattice.get_magnetization()
    energy = lattice.get_energy()

    return {
        "identity": identity,
        "magnetization": magnetization,
        "energy": energy,
        "magnitude": lattice.m,
        "size": lattice.size,
    }


def compute_identity_distance(
    identity_a: np.ndarray,
    identity_b: np.ndarray,
) -> float:
    """
    Compute angular distance between identity vectors.

    Uses cosine distance in the complex plane.

    Args:
        identity_a: First identity vector [2D]
        identity_b: Second identity vector [2D]

    Returns:
        Distance [0, 1] (0 = identical, 1 = opposite)
    """
    # Normalize vectors
    norm_a = np.linalg.norm(identity_a)
    norm_b = np.linalg.norm(identity_b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0  # Undefined if either is zero

    a_hat = identity_a / norm_a
    b_hat = identity_b / norm_b

    # Cosine similarity
    cos_sim = np.dot(a_hat, b_hat)

    # Convert to distance [0, 1]
    distance = (1.0 - cos_sim) / 2.0

    return np.clip(distance, 0.0, 1.0)


def extract_identity_tensor(
    lattice: TorsionLattice3D,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract full 3D phase tensor (for advanced analysis).

    Args:
        lattice: TorsionLattice3D instance
        normalize: If True, normalize phases to [0, 1]

    Returns:
        Phase tensor [size, size, size]
    """
    theta = np.copy(lattice.theta)

    if normalize:
        theta = theta / (2 * np.pi)

    return theta


def monitor_lattice(lattice: TorsionLattice3D) -> str:
    """
    Generate human-readable lattice status report.

    Args:
        lattice: TorsionLattice3D instance

    Returns:
        Formatted status string
    """
    stats = read_identity_with_stats(lattice)

    report = f"""
Torsion Lattice Status
{'=' * 50}
Size:          {stats['size']}³ = {stats['size']**3} nodes
Magnitude:     {stats['magnitude']:.6f}
Magnetization: {stats['magnetization']:.4f}
Energy:        {stats['energy']:.2f}

Identity Vector:
  Real:  {stats['identity'][0]: .6f}
  Imag:  {stats['identity'][1]: .6f}
  Norm:  {np.linalg.norm(stats['identity']):.6f}
{'=' * 50}
"""
    return report.strip()
