"""
EchoMirror Training - Hebbian learning without backpropagation.

This module implements resonance-based plasticity for coupling updates.
"""

from .echo_mirror import EchoMirrorTrainer, hebbian_update
from .datastream import MultimodalDatastream, create_datastream
from .loop import training_loop

__all__ = [
    "EchoMirrorTrainer",
    "hebbian_update",
    "MultimodalDatastream",
    "create_datastream",
    "training_loop",
]
