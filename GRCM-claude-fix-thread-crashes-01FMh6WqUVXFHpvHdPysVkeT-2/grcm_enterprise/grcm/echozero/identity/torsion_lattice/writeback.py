"""
Identity Writeback - Safe Memory Integration

Utilities for writing identity vectors into the 3D torsion lattice.
Enforces safety constraints (weak writes, Möbius gating).
"""

import numpy as np
from typing import Optional, Dict, Any
from .lattice3d import TorsionLattice3D


def write_identity(
    lattice: TorsionLattice3D,
    vector: np.ndarray,
    strength: float = 0.03,
) -> None:
    """
    Writes a projected identity vector into the 3D lattice.

    This is a simple wrapper around lattice.write_vector()
    for consistency with the module API.

    Args:
        lattice: TorsionLattice3D instance
        vector: Identity vector [2D] to write
        strength: Write strength coefficient (default: 0.03 < 0.05)
    """
    # Enforce safety: strength must be < 0.05
    safe_strength = min(strength, 0.05)

    lattice.write_vector(vector, strength=safe_strength)


def safe_write_with_validation(
    lattice: TorsionLattice3D,
    vector: np.ndarray,
    strength: float = 0.03,
    torsion_score: Optional[float] = None,
    torsion_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Write identity with Möbius gating and validation.

    Checks torsion score before writing. If torsion is too high,
    the write is rejected to preserve stability.

    Args:
        lattice: TorsionLattice3D instance
        vector: Identity vector [2D]
        strength: Write strength
        torsion_score: Möbius torsion (if None, write is always allowed)
        torsion_threshold: Max allowed torsion

    Returns:
        Dictionary with write status
    """
    # Check Möbius gate
    if torsion_score is not None and torsion_score > torsion_threshold:
        return {
            "written": False,
            "reason": "high_torsion",
            "torsion": torsion_score,
            "threshold": torsion_threshold,
        }

    # Validate vector
    if not isinstance(vector, np.ndarray) or vector.shape != (2,):
        return {
            "written": False,
            "reason": "invalid_vector_shape",
            "expected": (2,),
            "got": vector.shape if isinstance(vector, np.ndarray) else None,
        }

    # Check for NaN/Inf
    if not np.isfinite(vector).all():
        return {
            "written": False,
            "reason": "non_finite_values",
        }

    # Perform safe write
    write_identity(lattice, vector, strength=strength)

    return {
        "written": True,
        "strength": min(strength, 0.05),
        "vector_norm": np.linalg.norm(vector),
        "torsion": torsion_score,
    }


def incremental_write(
    lattice: TorsionLattice3D,
    target_vector: np.ndarray,
    num_steps: int = 10,
    total_strength: float = 0.03,
) -> None:
    """
    Write identity incrementally over multiple steps.

    Spreads a single write across multiple XY-model relaxation steps,
    allowing the lattice to adapt smoothly.

    Args:
        lattice: TorsionLattice3D instance
        target_vector: Identity vector [2D]
        num_steps: Number of incremental write steps
        total_strength: Total write strength (divided across steps)
    """
    step_strength = total_strength / num_steps

    for _ in range(num_steps):
        lattice.write_vector(target_vector, strength=step_strength)
        lattice.step()  # Relax after each write


def batch_write(
    lattice: TorsionLattice3D,
    vectors: np.ndarray,
    weights: Optional[np.ndarray] = None,
    total_strength: float = 0.03,
) -> Dict[str, Any]:
    """
    Write multiple identity vectors with optional weighting.

    Useful for consolidating multiple memory snapshots.

    Args:
        lattice: TorsionLattice3D instance
        vectors: Array of identity vectors [N, 2]
        weights: Optional weights [N] (if None, uniform)
        total_strength: Total write strength

    Returns:
        Dictionary with batch write statistics
    """
    N = vectors.shape[0]

    if weights is None:
        weights = np.ones(N) / N
    else:
        weights = weights / weights.sum()  # Normalize

    # Compute weighted average vector
    avg_vector = np.zeros(2, dtype=np.float64)
    for i in range(N):
        avg_vector += weights[i] * vectors[i]

    # Write averaged vector
    write_identity(lattice, avg_vector, strength=total_strength)

    return {
        "num_vectors": N,
        "avg_vector": avg_vector,
        "avg_norm": np.linalg.norm(avg_vector),
        "strength": total_strength,
    }
