"""
Tachyon Integration Wrapper

Connects the Tachyon pipeline to EchoZero architecture:
  Torsion Lattice → Tachyon Detector → Hopfield Classifier → Identity Vector

Key properties:
- Zero backreaction: Tachyon layer only reads from Torsion Lattice
- Non-blocking: Runs in parallel with Fast Loop
- Memory-safe: Soft updates only
- Modular: Can be disabled without affecting core pipeline
"""

import numpy as np
from typing import Dict, Optional
from .tachyon_detector import TachyonDetector
from .hopfield_classifier import HopfieldEventClassifier
from .identity_vector import IdentityVector


class TachyonIntegrationWrapper:
    """
    Full Tachyon pipeline integration.

    Usage:
        wrapper = TachyonIntegrationWrapper(num_patterns=20)

        # In EchoZero forward loop:
        result = wrapper.process_step(torsion_lattice.state)

        if result['event_detected']:
            pattern_idx = result['pattern_index']
            identity = result['identity_state']
    """

    def __init__(
        self,
        lattice_size: int = 5,
        num_patterns: int = 20,
        torsion_threshold: float = 2.5,
        enabled: bool = True
    ):
        """
        Initialize Tachyon integration wrapper.

        Args:
            lattice_size: Size of Torsion Lattice (5 for 5×5×5)
            num_patterns: Number of event pattern types
            torsion_threshold: Detection threshold for spike events
            enabled: Master enable flag
        """
        self.lattice_size = lattice_size
        self.num_patterns = num_patterns
        self.enabled = enabled

        # Initialize pipeline components
        self.detector = TachyonDetector(
            lattice_size=lattice_size,
            threshold=torsion_threshold
        )

        self.classifier = HopfieldEventClassifier(
            signature_dim=6,  # Event signatures are 6D
            max_patterns=num_patterns
        )

        self.identity = IdentityVector(
            num_patterns=num_patterns,
            decay_rate=0.01,
            update_strength=0.05
        )

        # Statistics
        self.total_steps = 0
        self.events_detected = 0
        self.last_event_step = 0

    def process_step(self, phase_field: np.ndarray) -> Dict:
        """
        Process one step of the Tachyon pipeline.

        Args:
            phase_field: Complex lattice state from Torsion Lattice (5×5×5)

        Returns:
            Dictionary with pipeline outputs
        """
        self.total_steps += 1

        if not self.enabled:
            return {
                'enabled': False,
                'event_detected': False
            }

        # Apply temporal decay to identity
        self.identity.decay()

        # Step 1: Detect torsion spike
        event = self.detector.detect(phase_field)

        if event is None:
            return {
                'enabled': True,
                'event_detected': False,
                'identity_state': self.identity.get_state()
            }

        # Step 2: Classify event signature
        signature = event['signature']
        pattern_idx, confidence = self.classifier.classify_direct(signature)

        # Step 3: Reinforce identity dimension
        if pattern_idx >= 0 and confidence > 0.5:
            self.identity.reinforce(pattern_idx, strength=confidence * 0.1)

        # Track statistics
        self.events_detected += 1
        self.last_event_step = self.total_steps

        return {
            'enabled': True,
            'event_detected': True,
            'event': event,
            'pattern_index': pattern_idx,
            'confidence': confidence,
            'identity_state': self.identity.get_state(),
            'identity_stats': self.identity.get_statistics()
        }

    def learn_prototype_from_events(
        self,
        num_samples: int = 10,
        label: Optional[str] = None
    ):
        """
        Learn a new event prototype from recent detections.

        Collects signatures from recent events and adds to classifier.
        """
        recent_events = self.detector.get_recent_events(n=num_samples)

        if len(recent_events) == 0:
            print("No recent events to learn from")
            return

        # Average signatures to create prototype
        signatures = [evt['signature'] for evt in recent_events]
        avg_signature = np.mean(signatures, axis=0)

        # Learn prototype
        pattern_idx = self.classifier.learn_prototype(avg_signature, label)

        return pattern_idx

    def batch_learn_prototypes(
        self,
        signatures: list,
        labels: Optional[list] = None
    ):
        """
        Learn multiple prototypes in batch (uses pseudo-inverse).

        Args:
            signatures: List of 6D signature vectors
            labels: Optional list of semantic labels
        """
        self.classifier.learn_prototypes_batch(signatures, labels)

    def get_statistics(self) -> Dict:
        """Get full pipeline statistics."""
        return {
            'total_steps': self.total_steps,
            'events_detected': self.events_detected,
            'last_event_step': self.last_event_step,
            'detection_rate': self.events_detected / max(1, self.total_steps),
            'prototypes_learned': len(self.classifier.prototypes),
            'identity': self.identity.get_statistics(),
            'enabled': self.enabled
        }

    def reset(self):
        """Reset all pipeline components."""
        self.detector.clear_history()
        self.classifier.reset()
        self.identity.reset()
        self.total_steps = 0
        self.events_detected = 0

    def enable(self):
        """Enable Tachyon pipeline."""
        self.enabled = True

    def disable(self):
        """Disable Tachyon pipeline (zero backreaction mode)."""
        self.enabled = False

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"TachyonIntegrationWrapper(enabled={self.enabled}, "
                f"events={stats['events_detected']}, "
                f"prototypes={stats['prototypes_learned']})")
