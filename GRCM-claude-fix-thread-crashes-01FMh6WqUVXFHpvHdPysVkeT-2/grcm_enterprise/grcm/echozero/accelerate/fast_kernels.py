"""
Fast Kernels for EchoZero - JIT-Accelerated Operations

Provides sub-millisecond Hopfield updates and torsion measurement.
Falls back to NumPy if JAX is unavailable.
"""

import numpy as np
import time

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class FastKernels:
    """
    JIT-accelerated kernels for critical operations.

    Provides:
    - Hopfield attractor step (W @ v normalization)
    - Torsion measurement (gradient-based)
    - Phase discontinuity detection

    Performance:
    - NumPy: ~1-2ms per operation
    - JAX (JIT): ~0.1-0.2ms per operation (after warmup)
    """

    def __init__(self, dim, enable_jax=True):
        self.dim = dim
        self.use_jax = JAX_AVAILABLE and enable_jax

        if self.use_jax:
            # JIT-compile functions
            self.hopfield_step = jax.jit(self._hopfield_step_jax)
            self.torsion_score = jax.jit(self._torsion_score_jax)
            self.phase_discontinuity = jax.jit(self._phase_discontinuity_jax)

            # Warmup JIT
            self._warmup()
        else:
            # Use NumPy implementations
            self.hopfield_step = self._hopfield_step_numpy
            self.torsion_score = self._torsion_score_numpy
            self.phase_discontinuity = self._phase_discontinuity_numpy

    def _warmup(self):
        """Warmup JIT compilation with dummy data."""
        dummy_v = jnp.ones(self.dim)
        dummy_W = jnp.eye(self.dim)

        # Trigger compilation
        _ = self.hopfield_step(dummy_v, dummy_W)
        _ = self.torsion_score(dummy_v)
        _ = self.phase_discontinuity(dummy_v)

    # =========================================================================
    # NUMPY IMPLEMENTATIONS (Fallback)
    # =========================================================================

    @staticmethod
    def _hopfield_step_numpy(v, W):
        """
        Single Hopfield attractor step: v_{t+1} = normalize(W @ v)

        Args:
            v: State vector (dim,)
            W: Weight matrix (dim, dim)

        Returns:
            Updated normalized vector
        """
        new = W @ v
        norm = np.linalg.norm(new)
        return new / (norm + 1e-8)

    @staticmethod
    def _torsion_score_numpy(v):
        """
        Rapid torsion estimator: L2 norm of first-order difference.

        This approximates the curl/torsion by measuring discontinuities.

        Args:
            v: State vector (dim,)

        Returns:
            Scalar torsion magnitude
        """
        return float(np.linalg.norm(np.diff(v)))

    @staticmethod
    def _phase_discontinuity_numpy(v):
        """
        Detect phase discontinuities (jumps > π).

        Args:
            v: Phase vector (dim,)

        Returns:
            Maximum phase jump magnitude
        """
        diff = np.diff(v)
        # Wrap to [-π, π]
        wrapped = (diff + np.pi) % (2 * np.pi) - np.pi
        return float(np.max(np.abs(wrapped)))

    # =========================================================================
    # JAX IMPLEMENTATIONS (JIT-Accelerated)
    # =========================================================================

    @staticmethod
    def _hopfield_step_jax(v, W):
        """JAX version of Hopfield step."""
        z = W @ v
        return z / (jnp.linalg.norm(z) + 1e-8)

    @staticmethod
    def _torsion_score_jax(v):
        """JAX version of torsion measurement."""
        return jnp.linalg.norm(jnp.diff(v))

    @staticmethod
    def _phase_discontinuity_jax(v):
        """JAX version of phase discontinuity detection."""
        diff = jnp.diff(v)
        wrapped = (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.max(jnp.abs(wrapped))

    # =========================================================================
    # BENCHMARKING
    # =========================================================================

    def benchmark(self, num_iterations=1000):
        """
        Benchmark performance of kernels.

        Returns:
            dict with timing results
        """
        v = np.random.randn(self.dim)
        W = np.random.randn(self.dim, self.dim)

        if self.use_jax:
            v_jax = jnp.array(v)
            W_jax = jnp.array(W)

        # Hopfield step benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            if self.use_jax:
                _ = self.hopfield_step(v_jax, W_jax)
            else:
                _ = self.hopfield_step(v, W)
        hopfield_time = (time.perf_counter() - start) / num_iterations

        # Torsion score benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            if self.use_jax:
                _ = self.torsion_score(v_jax)
            else:
                _ = self.torsion_score(v)
        torsion_time = (time.perf_counter() - start) / num_iterations

        return {
            'backend': 'JAX' if self.use_jax else 'NumPy',
            'dim': self.dim,
            'hopfield_step_ms': hopfield_time * 1000,
            'torsion_score_ms': torsion_time * 1000,
            'total_ms': (hopfield_time + torsion_time) * 1000
        }

    def get_info(self):
        """Get kernel information."""
        return {
            'backend': 'JAX' if self.use_jax else 'NumPy',
            'dim': self.dim,
            'jax_available': JAX_AVAILABLE,
        }
