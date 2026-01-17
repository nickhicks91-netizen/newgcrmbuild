"""
Lattice Updater - Background Synchronization

Periodically integrates EchoZero/GRCM state into the 3D torsion lattice.
Operates on a SLOW LOOP (0.1-2 Hz) that never blocks fast inference.

Key Features:
- Periodic sync (every N steps)
- Möbius-gated writes (only stable patterns)
- Background relaxation (self-healing)
- Non-blocking (fast loop continues independently)
"""

import numpy as np
from typing import Optional, Dict, Any
from .lattice3d import TorsionLattice3D


class LatticeUpdater:
    """
    Periodically integrates EchoCore state into 3D lattice.

    This implements the SLOW LOOP that:
    1. Samples fast loop state every N steps
    2. Filters by Möbius torsion (stability check)
    3. Writes only low-torsion patterns
    4. Relaxes lattice toward stable minima
    """

    def __init__(
        self,
        lattice: TorsionLattice3D,
        write_strength: float = 0.12,
        sync_interval: int = 300,
        torsion_threshold: float = 2.5,
        relaxation_steps: int = 40,
    ):
        """
        Initialize lattice updater.

        Args:
            lattice: TorsionLattice3D instance to update
            write_strength: Write coefficient (0.12 for good retention)
            sync_interval: Steps between sync attempts (controls Hz)
            torsion_threshold: Max torsion for writes (2.5 for realistic high-dim data)
            relaxation_steps: XY-model steps per sync (40 for deep healing)

        Physics regime (post-patch):
            write_strength = 0.12    # meaningful write influence
            torsion_threshold = 2.5  # realistic for N-dimensional states (scales ~sqrt(N))
            relaxation_steps = 40    # deeper healing sweep
        """
        self.lattice = lattice
        self.write_strength = write_strength
        self.sync_interval = sync_interval
        self.torsion_threshold = torsion_threshold
        self.relaxation_steps = relaxation_steps

        # Internal state
        self.step_count = 0
        self.total_syncs = 0
        self.rejected_syncs = 0
        self.last_torsion = 0.0

    def maybe_sync(
        self,
        echo_state: np.ndarray,
        torsion_score: float,
    ) -> Dict[str, Any]:
        """
        Conditionally sync EchoZero state to lattice.

        This is called every fast-loop step, but only commits
        to the lattice every sync_interval steps if torsion is low.

        Args:
            echo_state: Identity vector from EchoCore [2D]
            torsion_score: Möbius torsion score (low = stable)

        Returns:
            Dictionary with sync statistics
        """
        self.step_count += 1
        self.last_torsion = torsion_score

        # Check if it's time to attempt sync
        if self.step_count % self.sync_interval != 0:
            return {
                "synced": False,
                "reason": "not_interval",
                "next_sync_in": self.sync_interval - (self.step_count % self.sync_interval),
            }

        # Möbius stability gate: only commit low-torsion states
        if torsion_score > self.torsion_threshold:
            self.rejected_syncs += 1
            return {
                "synced": False,
                "reason": "high_torsion",
                "torsion": torsion_score,
                "threshold": self.torsion_threshold,
                "rejected_count": self.rejected_syncs,
            }

        # WRITE: Commit state to lattice (weak write)
        self.lattice.write_vector(echo_state, strength=self.write_strength)

        # RELAX: Run XY-model dynamics to heal and stabilize
        for _ in range(self.relaxation_steps):
            self.lattice.step()

        self.total_syncs += 1

        return {
            "synced": True,
            "torsion": torsion_score,
            "sync_count": self.total_syncs,
            "magnetization": self.lattice.get_magnetization(),
            "energy": self.lattice.get_energy(),
        }

    def force_sync(
        self,
        echo_state: np.ndarray,
        skip_torsion_check: bool = False,
    ) -> Dict[str, Any]:
        """
        Force immediate sync (for testing/debugging).

        Args:
            echo_state: Identity vector [2D]
            skip_torsion_check: If True, bypass Möbius gate

        Returns:
            Sync statistics
        """
        # Write and relax
        self.lattice.write_vector(echo_state, strength=self.write_strength)

        for _ in range(self.relaxation_steps):
            self.lattice.step()

        self.total_syncs += 1

        return {
            "synced": True,
            "forced": True,
            "sync_count": self.total_syncs,
            "magnetization": self.lattice.get_magnetization(),
            "energy": self.lattice.get_energy(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get updater statistics.

        Returns:
            Dictionary with sync metrics
        """
        acceptance_rate = (
            self.total_syncs / max(1, self.total_syncs + self.rejected_syncs)
        )

        return {
            "step_count": self.step_count,
            "total_syncs": self.total_syncs,
            "rejected_syncs": self.rejected_syncs,
            "acceptance_rate": acceptance_rate,
            "last_torsion": self.last_torsion,
            "sync_interval": self.sync_interval,
            "write_strength": self.write_strength,
            "lattice_magnetization": self.lattice.get_magnetization(),
            "lattice_energy": self.lattice.get_energy(),
        }

    def reset(self) -> None:
        """Reset updater statistics (not lattice)."""
        self.step_count = 0
        self.total_syncs = 0
        self.rejected_syncs = 0
        self.last_torsion = 0.0

    def reset_all(self) -> None:
        """Reset both updater and lattice."""
        self.reset()
        self.lattice.reset()

    def set_sync_interval(self, interval: int) -> None:
        """
        Adjust sync frequency.

        Args:
            interval: New sync interval (steps)
        """
        self.sync_interval = max(1, interval)

    def set_torsion_threshold(self, threshold: float) -> None:
        """
        Adjust Möbius gate threshold.

        Args:
            threshold: New torsion threshold [0, 1]
        """
        self.torsion_threshold = np.clip(threshold, 0.0, 1.0)

    def __repr__(self) -> str:
        """String representation of updater state."""
        stats = self.get_stats()
        return (
            f"LatticeUpdater("
            f"syncs={stats['total_syncs']}, "
            f"rejected={stats['rejected_syncs']}, "
            f"acceptance={stats['acceptance_rate']:.2%}, "
            f"interval={self.sync_interval})"
        )
