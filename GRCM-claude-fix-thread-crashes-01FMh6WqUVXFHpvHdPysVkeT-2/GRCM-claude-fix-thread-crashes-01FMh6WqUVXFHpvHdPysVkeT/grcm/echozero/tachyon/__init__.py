"""
Tachyon Salience Layer for EchoZero

Event-driven identity management via torsion spike detection.

Architecture:
    Torsion Lattice (XY-model)
           ↓
    Tachyon Detector (spike detection)
           ↓
    Hopfield Classifier (pattern indexing)
           ↓
    Identity Vector (sparse updates)

Usage:
    from grcm.echozero.tachyon import TachyonIntegrationWrapper

    tachyon = TachyonIntegrationWrapper(num_patterns=20)

    # In forward loop:
    result = tachyon.process_step(torsion_lattice.state)
"""

from .tachyon_detector import TachyonDetector
from .hopfield_classifier import HopfieldEventClassifier
from .identity_vector import IdentityVector
from .integration_wrapper import TachyonIntegrationWrapper

__all__ = [
    'TachyonDetector',
    'HopfieldEventClassifier',
    'IdentityVector',
    'TachyonIntegrationWrapper'
]
