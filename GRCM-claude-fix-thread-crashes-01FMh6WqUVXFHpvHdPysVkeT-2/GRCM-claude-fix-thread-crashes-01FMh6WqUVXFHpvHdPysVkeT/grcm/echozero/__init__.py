"""
EchoZero: Complex-valued resonant dynamics module.

This module implements coupled nonlinear oscillator dynamics with:
- Local dynamics (-α + iω)ψ
- Nearest-neighbor coupling
- Nonlinear damping -β|ψ|²ψ
- Want modulation γψ
- External drive I(t)
- Hub constraint -λΣψ
- M\u00f6bius topological consistency enforcement
"""

from .dynamics import (
    echozero_dynamics,
    EchoZeroSystem,
    compute_coherence,
    compute_want_modulation,
)
from .lattice import LatticeBuilder, build_ring_lattice
from .coupling import CouplingMatrix, build_coupling_matrix
from .ode_solver import integrate_echozero, RK4Solver
from .mobius import MobiusEchoLayer, create_mobius_layer
from .spiral import SpiralLattice, create_spiral_lattice
from .memory_engine import EchoZeroMemoryEngine, EchoZeroMemoryEngineWithSnapback
from .memory_engine_threadsafe import (
    ThreadSafeEchoZeroMemoryEngine,
    ThreadSafeEchoZeroMemoryEngineWithSnapback,
)

__all__ = [
    "echozero_dynamics",
    "EchoZeroSystem",
    "compute_coherence",
    "compute_want_modulation",
    "LatticeBuilder",
    "build_ring_lattice",
    "CouplingMatrix",
    "build_coupling_matrix",
    "integrate_echozero",
    "RK4Solver",
    "MobiusEchoLayer",
    "create_mobius_layer",
    "SpiralLattice",
    "create_spiral_lattice",
    "EchoZeroMemoryEngine",
    "EchoZeroMemoryEngineWithSnapback",
    "ThreadSafeEchoZeroMemoryEngine",
    "ThreadSafeEchoZeroMemoryEngineWithSnapback",
]
