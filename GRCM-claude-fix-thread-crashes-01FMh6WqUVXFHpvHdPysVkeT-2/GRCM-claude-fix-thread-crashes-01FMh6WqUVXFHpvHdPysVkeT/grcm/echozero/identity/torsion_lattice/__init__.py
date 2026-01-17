"""
3D Torsion Memory Lattice - Long-term Identity Storage

This module provides self-healing, noise-resistant long-term memory
for the EchoZero/GRCM hybrid system.

Architecture:
- SLOW LOOP (0.1-2 Hz): Torsion lattice consolidation
- FAST LOOP (1-10ms): EchoZero inference (unchanged)

Key Features:
- Modular: Drop-in addition to existing system
- Safe: Weak writes (α < 0.05), Möbius-gated
- Non-invasive: Never blocks fast loop
- Self-healing: XY-model relaxation dynamics
- Production-ready: 100% backward compatible

Usage:
    from grcm.echozero.identity.torsion_lattice import (
        TorsionLattice3D,
        LatticeUpdater,
        read_identity,
        write_identity,
    )

    # Create lattice
    lattice = TorsionLattice3D(size=5)

    # Create updater (slow loop manager)
    updater = LatticeUpdater(
        lattice=lattice,
        sync_interval=300,  # Sync every 300 steps
        torsion_threshold=0.2,  # Möbius gate
    )

    # In fast loop:
    for step in range(1000):
        # ... EchoZero/GRCM inference ...

        # Periodic sync (non-blocking)
        updater.maybe_sync(
            echo_state=identity_vector,  # [2D]
            torsion_score=mobius_torsion,  # float
        )

    # Read long-term identity
    identity = read_identity(lattice)
"""

from .lattice3d import TorsionLattice3D
from .update import LatticeUpdater
from .readout import (
    read_identity,
    read_identity_with_stats,
    compute_identity_distance,
    extract_identity_tensor,
    monitor_lattice,
)
from .writeback import (
    write_identity,
    safe_write_with_validation,
    incremental_write,
    batch_write,
)

__all__ = [
    # Core classes
    "TorsionLattice3D",
    "LatticeUpdater",

    # Readout functions
    "read_identity",
    "read_identity_with_stats",
    "compute_identity_distance",
    "extract_identity_tensor",
    "monitor_lattice",

    # Writeback functions
    "write_identity",
    "safe_write_with_validation",
    "incremental_write",
    "batch_write",
]

__version__ = "1.0.0"
