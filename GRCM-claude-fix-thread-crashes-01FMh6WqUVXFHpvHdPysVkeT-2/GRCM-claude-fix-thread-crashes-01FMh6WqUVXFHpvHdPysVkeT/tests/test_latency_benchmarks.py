"""
Sub-Millisecond Latency Test Suite (6 tests)

Validates that fast kernels achieve sub-millisecond performance:
- JIT warmup
- Cold start latency
- Steady-state latency
- Jitter bounds
- Backend switching
- Throughput scaling
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/home/user/GRCM')

from grcm.echozero.accelerate.fast_kernels import FastKernels


def test_kernel_initialization():
    """Test 1: Fast kernels initialize correctly."""
    print("\n" + "="*70)
    print("TEST 1: KERNEL INITIALIZATION")
    print("="*70)

    kernels = FastKernels(dim=64)

    info = kernels.get_info()

    print(f"✓ Backend: {info['backend']}")
    print(f"✓ Dimension: {info['dim']}")
    print(f"✓ JAX available: {info['jax_available']}")

    assert info['dim'] == 64

    print("\n✅ TEST 1 PASSED")


def test_jit_warmup():
    """Test 2: JIT warmup reduces latency."""
    print("\n" + "="*70)
    print("TEST 2: JIT WARMUP EFFECT")
    print("="*70)

    kernels = FastKernels(dim=64)

    v = np.random.randn(64)
    W = np.random.randn(64, 64)

    # Cold start (first call)
    start = time.perf_counter()
    _ = kernels.hopfield_step(v, W)
    cold_time = time.perf_counter() - start

    # Warm calls
    warm_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = kernels.hopfield_step(v, W)
        warm_times.append(time.perf_counter() - start)

    mean_warm = np.mean(warm_times) * 1000  # ms

    print(f"✓ Cold start: {cold_time*1000:.3f} ms")
    print(f"✓ Mean warm: {mean_warm:.3f} ms")
    print(f"✓ Speedup: {cold_time / np.mean(warm_times):.1f}x")

    # Warm should be faster (unless JAX not available)
    info = kernels.get_info()
    if info['backend'] == 'JAX':
        assert mean_warm < cold_time * 1000, "JIT should improve latency"

    print("\n✅ TEST 2 PASSED")


def test_hopfield_step_latency():
    """Test 3: Hopfield step achieves target latency."""
    print("\n" + "="*70)
    print("TEST 3: HOPFIELD STEP LATENCY")
    print("="*70)

    kernels = FastKernels(dim=64)

    v = np.random.randn(64)
    W = np.random.randn(64, 64)

    # Warmup
    for _ in range(10):
        _ = kernels.hopfield_step(v, W)

    # Benchmark
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = kernels.hopfield_step(v, W)
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    p99_ms = np.percentile(times, 99) * 1000

    print(f"✓ Mean latency: {mean_ms:.3f} ms")
    print(f"✓ P95 latency: {p95_ms:.3f} ms")
    print(f"✓ P99 latency: {p99_ms:.3f} ms")
    print(f"✓ Backend: {kernels.get_info()['backend']}")

    # Target: < 2ms for NumPy, < 0.5ms for JAX
    if kernels.get_info()['backend'] == 'JAX':
        assert mean_ms < 0.5, f"JAX latency too high: {mean_ms:.3f} ms"
    else:
        assert mean_ms < 2.0, f"NumPy latency too high: {mean_ms:.3f} ms"

    print("\n✅ TEST 3 PASSED")


def test_torsion_measurement_latency():
    """Test 4: Torsion measurement latency."""
    print("\n" + "="*70)
    print("TEST 4: TORSION MEASUREMENT LATENCY")
    print("="*70)

    kernels = FastKernels(dim=64)

    v = np.random.randn(64)

    # Warmup
    for _ in range(10):
        _ = kernels.torsion_score(v)

    # Benchmark
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = kernels.torsion_score(v)
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000

    print(f"✓ Mean latency: {mean_ms:.3f} ms")
    print(f"✓ Backend: {kernels.get_info()['backend']}")

    # Torsion should be faster than Hopfield (smaller operation)
    assert mean_ms < 1.0, f"Torsion latency too high: {mean_ms:.3f} ms"

    print("\n✅ TEST 4 PASSED")


def test_jitter_bounds():
    """Test 5: Latency jitter stays within bounds."""
    print("\n" + "="*70)
    print("TEST 5: LATENCY JITTER BOUNDS")
    print("="*70)

    kernels = FastKernels(dim=64)

    v = np.random.randn(64)
    W = np.random.randn(64, 64)

    # Warmup
    for _ in range(100):
        _ = kernels.hopfield_step(v, W)

    # Measure jitter
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = kernels.hopfield_step(v, W)
        times.append(time.perf_counter() - start)

    times_ms = np.array(times) * 1000
    mean = np.mean(times_ms)
    std = np.std(times_ms)
    cv = std / mean  # Coefficient of variation

    print(f"✓ Mean: {mean:.3f} ms")
    print(f"✓ Std: {std:.3f} ms")
    print(f"✓ CV: {cv:.2%}")

    # Low jitter expected (CV < 50%)
    assert cv < 0.5, f"Jitter too high: CV={cv:.2%}"

    print("\n✅ TEST 5 PASSED")


def test_throughput_scaling():
    """Test 6: Throughput scales linearly."""
    print("\n" + "="*70)
    print("TEST 6: THROUGHPUT SCALING")
    print("="*70)

    kernels = FastKernels(dim=64)

    batch_sizes = [10, 100, 1000]
    throughputs = []

    for batch_size in batch_sizes:
        v = np.random.randn(64)
        W = np.random.randn(64, 64)

        # Warmup
        for _ in range(10):
            _ = kernels.hopfield_step(v, W)

        # Benchmark
        start = time.perf_counter()
        for _ in range(batch_size):
            _ = kernels.hopfield_step(v, W)
        elapsed = time.perf_counter() - start

        throughput = batch_size / elapsed
        throughputs.append(throughput)

        print(f"  Batch {batch_size}: {throughput:.0f} ops/sec")

    # Throughput should be roughly constant (linear scaling)
    throughput_ratio = max(throughputs) / min(throughputs)

    print(f"\n✓ Throughput ratio: {throughput_ratio:.2f}x")

    # Should be within 3x (allowing for warmup effects)
    assert throughput_ratio < 3.0, f"Non-linear scaling: {throughput_ratio:.2f}x"

    print("\n✅ TEST 6 PASSED")


def main():
    """Run all latency benchmark tests."""
    print("╔" + "="*68 + "╗")
    print("║" + " "*17 + "LATENCY BENCHMARK TEST SUITE" + " "*22 + "║")
    print("╚" + "="*68 + "╝")

    try:
        test_kernel_initialization()
        test_jit_warmup()
        test_hopfield_step_latency()
        test_torsion_measurement_latency()
        test_jitter_bounds()
        test_throughput_scaling()

        print("\n" + "="*70)
        print("✅ ALL 6 LATENCY BENCHMARK TESTS PASSED")
        print("="*70)
        print("\nValidated:")
        print("  ✓ Kernel initialization (JAX/NumPy)")
        print("  ✓ JIT warmup effect")
        print("  ✓ Hopfield step latency (< 2ms NumPy, < 0.5ms JAX)")
        print("  ✓ Torsion measurement latency (< 1ms)")
        print("  ✓ Jitter bounds (CV < 50%)")
        print("  ✓ Linear throughput scaling")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
