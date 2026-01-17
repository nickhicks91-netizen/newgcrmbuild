"""
Hardware mapping for EchoZero.

Maps abstract dynamics to physical substrates:
- Photonic: Silicon nitride microring resonators
- Magnonic: YIG spin wave networks
"""

from .photonic_map import PhotonicMapper, map_to_photonic
from .magnonic_map import MagnonicMapper, map_to_magnonic

__all__ = [
    "PhotonicMapper",
    "map_to_photonic",
    "MagnonicMapper",
    "map_to_magnonic",
]
