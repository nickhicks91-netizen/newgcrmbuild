"""
Tachyon Spike Detector

Monitors the Torsion Lattice (XY-model) for discontinuities and phase jumps,
producing discrete "memory scar" events that trigger identity updates.

Architecture:
- Input: Phase field from Torsion Lattice (5×5×5 complex grid)
- Output: Discrete spike events with signature vectors
- No reconstruction: outputs event descriptors only

Each event contains:
- Torsion magnitude (peak curvature)
- Phase jump (discontinuity measure)
- Spatial location
- Temporal signature
"""

import numpy as np
from typing import Dict, List, Optional


class TachyonDetector:
    """
    Detects torsion spikes from phase field discontinuities.

    Monitors the Torsion Lattice XY-model for:
    - Phase discontinuities (∇θ spikes)
    - Energy density peaks
    - Coherence breaks

    Produces discrete event signatures for Hopfield classification.
    """

    def __init__(
        self,
        lattice_size: int = 5,
        threshold: float = 2.5,
        history_length: int = 100
    ):
        """
        Initialize tachyon detector.

        Args:
            lattice_size: Size of input lattice (5 for 5×5×5)
            threshold: Torsion magnitude threshold for event detection
            history_length: Number of recent events to track
        """
        self.lattice_size = lattice_size
        self.threshold = threshold
        self.history_length = history_length

        # Event memory
        self.events = []  # List of detected events
        self.event_count = 0

        # Temporal context
        self.last_phase_field = None
        self.step_count = 0

    def detect(self, phase_field: np.ndarray) -> Optional[Dict]:
        """
        Detect torsion spikes in current phase field.

        Args:
            phase_field: Complex lattice state (size, size, size)

        Returns:
            Event dict if spike detected, None otherwise
        """
        self.step_count += 1

        # Convert complex to phases
        phases = np.angle(phase_field)

        # Compute torsion metrics
        torsion_magnitude = self._compute_torsion(phases)
        phase_jump = self._compute_phase_jump(phases)

        # Check for spike event
        if torsion_magnitude > self.threshold:
            event = self._create_event(
                torsion_magnitude,
                phase_jump,
                phases
            )

            self.events.append(event)
            self.event_count += 1

            # Keep history bounded
            if len(self.events) > self.history_length:
                self.events.pop(0)

            self.last_phase_field = phases.copy()
            return event

        self.last_phase_field = phases.copy()
        return None

    def _compute_torsion(self, phases: np.ndarray) -> float:
        """
        Compute torsion magnitude from phase field.

        Uses discrete curl to measure phase winding.
        """
        # Compute phase gradients (discrete derivatives)
        grad_x = np.roll(phases, -1, axis=0) - phases
        grad_y = np.roll(phases, -1, axis=1) - phases
        grad_z = np.roll(phases, -1, axis=2) - phases

        # Wrap gradients to [-π, π]
        grad_x = (grad_x + np.pi) % (2 * np.pi) - np.pi
        grad_y = (grad_y + np.pi) % (2 * np.pi) - np.pi
        grad_z = (grad_z + np.pi) % (2 * np.pi) - np.pi

        # Compute curl magnitude (discrete)
        curl_x = np.roll(grad_z, -1, axis=1) - np.roll(grad_y, -1, axis=2)
        curl_y = np.roll(grad_x, -1, axis=2) - np.roll(grad_z, -1, axis=0)
        curl_z = np.roll(grad_y, -1, axis=0) - np.roll(grad_x, -1, axis=1)

        curl_magnitude = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)

        # Return peak torsion
        return np.max(curl_magnitude)

    def _compute_phase_jump(self, phases: np.ndarray) -> float:
        """
        Measure maximum phase discontinuity.

        Large jumps indicate topological defects.
        """
        if self.last_phase_field is None:
            return 0.0

        # Phase difference from previous step
        phase_diff = phases - self.last_phase_field

        # Wrap to [-π, π]
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

        # Return maximum jump
        return np.max(np.abs(phase_diff))

    def _create_event(
        self,
        torsion_magnitude: float,
        phase_jump: float,
        phases: np.ndarray
    ) -> Dict:
        """
        Create event descriptor from spike detection.

        Returns compact signature for Hopfield classification.
        """
        # Find spike location
        grad_x = np.roll(phases, -1, axis=0) - phases
        grad_y = np.roll(phases, -1, axis=1) - phases
        grad_z = np.roll(phases, -1, axis=2) - phases

        grad_x = (grad_x + np.pi) % (2 * np.pi) - np.pi
        grad_y = (grad_y + np.pi) % (2 * np.pi) - np.pi
        grad_z = (grad_z + np.pi) % (2 * np.pi) - np.pi

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        peak_location = np.unravel_index(
            np.argmax(gradient_magnitude),
            gradient_magnitude.shape
        )

        # Create event signature vector (for Hopfield input)
        # 6D signature: [torsion, phase_jump, x, y, z, temporal_phase]
        signature = np.array([
            torsion_magnitude / 10.0,  # Normalize
            phase_jump / np.pi,
            peak_location[0] / self.lattice_size,
            peak_location[1] / self.lattice_size,
            peak_location[2] / self.lattice_size,
            (self.step_count % 100) / 100.0  # Temporal context
        ])

        event = {
            'event_id': self.event_count,
            'step': self.step_count,
            'torsion_magnitude': torsion_magnitude,
            'phase_jump': phase_jump,
            'location': peak_location,
            'signature': signature,  # For Hopfield classification
            'timestamp': self.step_count
        }

        return event

    def get_recent_events(self, n: int = 10) -> List[Dict]:
        """Get n most recent events."""
        return self.events[-n:]

    def clear_history(self):
        """Clear event history."""
        self.events = []
        self.event_count = 0

    def __repr__(self) -> str:
        return (f"TachyonDetector(threshold={self.threshold:.2f}, "
                f"events_detected={self.event_count})")
