"""
EchoZero Acceleration Layer

Provides JIT-accelerated kernels for sub-millisecond performance.
"""

from .fast_kernels import FastKernels

__all__ = ['FastKernels']
