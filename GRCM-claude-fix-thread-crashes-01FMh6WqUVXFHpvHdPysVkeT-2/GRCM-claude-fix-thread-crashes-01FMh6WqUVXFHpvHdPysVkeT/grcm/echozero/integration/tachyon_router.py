"""
Tachyon Event Router

Routes torsion spike events to Hopfield network, spatial decoder,
and replay buffer. This is the "glue" layer that coordinates:

- Event detection (torsion threshold)
- Hopfield memory updates
- Spatial code storage
- Exact state logging

This module solves the "routing problem" identified in integration testing.
"""

import numpy as np
from typing import Tuple, Optional


class TachyonRouter:
    """
    Routes Tachyon torsion events through the memory system.

    Pipeline:
    1. Measure torsion
    2. If torsion > threshold → SYNC EVENT
    3. Update Hopfield state via fast path
    4. Store spatial code in decoder
    5. Log exact state in replay buffer

    This ensures:
    - Memory only updates on salient events (torsion spikes)
    - Spatial and exact reconstruction stay in sync
    - Zero backreaction (read-only until sync)

    Example:
        >>> router = TachyonRouter(hopfield, decoder, fastpath, replay)
        >>> new_state, synced, torsion = router.process(current_state)
    """

    def __init__(self,
                 hopfield,
                 decoder,
                 fastpath,
                 replay,
                 torsion_threshold=2.0):
        """
        Initialize Tachyon router.

        Args:
            hopfield: Hopfield network instance
            decoder: SpatialMemoryDecoder instance
            fastpath: FastKernels instance
            replay: ReplayBuffer instance
            torsion_threshold: Torsion magnitude threshold for sync events
        """
        self.hopfield = hopfield
        self.decoder = decoder
        self.fast = fastpath
        self.replay = replay

        self.torsion_threshold = torsion_threshold

        # Statistics
        self.total_steps = 0
        self.sync_events = 0
        self.rejected_syncs = 0  # High torsion rejections

    def process(self, state: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Process one state through Tachyon routing.

        Args:
            state: Current state vector (dim,)

        Returns:
            (updated_state, sync_flag, torsion_magnitude)

        Flow:
            1. Compute torsion
            2. Log to replay buffer (always)
            3. If torsion < threshold:
                - No sync
                - Return state unchanged
            4. If torsion >= threshold:
                - SYNC EVENT
                - Update Hopfield
                - Store spatial code
                - Return updated state
        """
        self.total_steps += 1

        # -----------------------------------------------------------------
        # 1. MEASURE TORSION
        # -----------------------------------------------------------------
        torsion = self.fast.torsion_score(state)

        # -----------------------------------------------------------------
        # 2. COMPUTE PHASE GRADIENT (for replay buffer metadata)
        # -----------------------------------------------------------------
        phase_grad = float(np.sum(np.abs(np.diff(state))))

        # -----------------------------------------------------------------
        # 3. ALWAYS LOG TO REPLAY BUFFER (exact reconstruction)
        # -----------------------------------------------------------------
        self.replay.push(state, torsion=torsion, phase_grad=phase_grad)

        # -----------------------------------------------------------------
        # 4. CHECK TORSION THRESHOLD
        # -----------------------------------------------------------------
        if torsion < self.torsion_threshold:
            # LOW TORSION → NO SYNC (coherent state, no update needed)
            return state, False, torsion

        # -----------------------------------------------------------------
        # 5. TACHYON SYNC EVENT → UPDATE MEMORY
        # -----------------------------------------------------------------
        self.sync_events += 1

        # Update Hopfield via fast path
        if hasattr(self.hopfield, 'W'):
            # Hopfield has weight matrix → use fast kernel
            new_state = self.fast.hopfield_step(state, self.hopfield.W)

            # Get attractor index (if Hopfield tracks it)
            if hasattr(self.hopfield, 'last_index'):
                idx = self.hopfield.last_index
            else:
                idx = None

        else:
            # Hopfield doesn't expose W → use native step
            if hasattr(self.hopfield, 'step'):
                new_state = self.hopfield.step(state)
            else:
                new_state = state  # Fallback: no update

            idx = None

        # -----------------------------------------------------------------
        # 6. STORE SPATIAL CODE (approximate reconstruction)
        # -----------------------------------------------------------------
        if idx is not None and self.decoder.fitted:
            self.decoder.store_attractor(idx, new_state)

        return new_state, True, torsion

    def get_statistics(self):
        """Get routing statistics."""
        return {
            'total_steps': self.total_steps,
            'sync_events': self.sync_events,
            'sync_rate': self.sync_events / max(self.total_steps, 1),
            'rejected_syncs': self.rejected_syncs,
            'torsion_threshold': self.torsion_threshold,
        }

    def reset_statistics(self):
        """Reset routing counters."""
        self.total_steps = 0
        self.sync_events = 0
        self.rejected_syncs = 0

    def __repr__(self):
        return (f"TachyonRouter(threshold={self.torsion_threshold}, "
                f"steps={self.total_steps}, syncs={self.sync_events})")
