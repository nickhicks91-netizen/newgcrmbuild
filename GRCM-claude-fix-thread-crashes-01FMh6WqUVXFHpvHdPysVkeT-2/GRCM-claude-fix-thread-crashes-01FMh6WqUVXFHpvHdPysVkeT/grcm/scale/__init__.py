"""
Scaling module for EchoZero.

Supports lattices from 64 nodes to 1B+ nodes.
"""

from .builder import ScalableLatticeBuilder, build_scaled_system
from .profiler import profile_system, benchmark_coherence

__all__ = [
    "ScalableLatticeBuilder",
    "build_scaled_system",
    "profile_system",
    "benchmark_coherence",
]
