"""
Comprehensive stress test suite for EchoZero + GRCM.

Tests:
- Forward pass performance
- Integration stability
- Training speed
- Scalability
- Memory usage
- Edge cases
"""

import time
import numpy as np
from typing import Dict, List
import traceback
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StressTestResults:
    """Container for stress test results."""

    def __init__(self):
        self.tests = {}
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_result(self, name: str, passed: bool, data: Dict = None, error: str = None):
        """Add a test result."""
        self.tests[name] = {
            "passed": passed,
            "data": data or {},
            "error": error,
        }
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            if error:
                self.errors.append((name, error))

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("  STRESS TEST SUMMARY")
        print("=" * 80)
        print(f"Total tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Success rate: {100 * self.passed / (self.passed + self.failed):.1f}%")

        if self.errors:
            print("\nErrors:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")


def test_imports():
    """Test that all modules can be imported."""
    print("\n[1/8] Testing module imports...")
    results = StressTestResults()

    try:
        # Core imports
        from grcm.echozero import (
            echozero_dynamics,
            EchoZeroSystem,
            compute_coherence,
            compute_want_modulation,
            build_ring_lattice,
            build_coupling_matrix,
            integrate_echozero,
        )
        results.add_result("echozero_imports", True)
        print("  ✅ EchoZero dynamics")
    except Exception as e:
        results.add_result("echozero_imports", False, error=str(e))
        print(f"  ❌ EchoZero dynamics: {e}")
        return results

    try:
        from grcm.hybrid import EchoGRCMHybrid
        results.add_result("hybrid_imports", True)
        print("  ✅ Hybrid module")
    except Exception as e:
        results.add_result("hybrid_imports", False, error=str(e))
        print(f"  ❌ Hybrid module: {e}")

    try:
        from grcm.train import EchoMirrorTrainer, MultimodalDatastream
        results.add_result("train_imports", True)
        print("  ✅ Training module")
    except Exception as e:
        results.add_result("train_imports", False, error=str(e))
        print(f"  ❌ Training module: {e}")

    try:
        from grcm.scale import ScalableLatticeBuilder
        results.add_result("scale_imports", True)
        print("  ✅ Scaling module")
    except Exception as e:
        results.add_result("scale_imports", False, error=str(e))
        print(f"  ❌ Scaling module: {e}")

    try:
        from grcm.hardware import PhotonicMapper, MagnonicMapper
        results.add_result("hardware_imports", True)
        print("  ✅ Hardware module")
    except Exception as e:
        results.add_result("hardware_imports", False, error=str(e))
        print(f"  ❌ Hardware module: {e}")

    return results


def test_forward_pass_performance(n_trials=100):
    """Benchmark forward pass performance."""
    print(f"\n[2/8] Testing forward pass performance ({n_trials} trials)...")
    results = StressTestResults()

    try:
        import torch
        from grcm.hybrid import EchoGRCMHybrid

        # Initialize model
        model = EchoGRCMHybrid(n_nodes=64)

        # Prepare inputs
        image_emb = torch.randn(1, 512)
        audio_emb = torch.randn(1, 768)
        action = torch.zeros(1, 4)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(image_emb, audio_emb, action)

        # Benchmark
        latencies = []
        start_total = time.time()

        for _ in range(n_trials):
            start = time.time()
            with torch.no_grad():
                outputs = model(image_emb, audio_emb, action)
            latencies.append(time.time() - start)

        total_time = time.time() - start_total

        # Statistics
        latencies_ms = np.array(latencies) * 1000
        data = {
            "mean_latency_ms": float(np.mean(latencies_ms)),
            "median_latency_ms": float(np.median(latencies_ms)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
            "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
            "throughput_samples_per_sec": n_trials / total_time,
            "n_trials": n_trials,
        }

        print(f"  Mean latency: {data['mean_latency_ms']:.2f}ms")
        print(f"  P95 latency: {data['p95_latency_ms']:.2f}ms")
        print(f"  P99 latency: {data['p99_latency_ms']:.2f}ms")
        print(f"  Throughput: {data['throughput_samples_per_sec']:.1f} samples/sec")

        # Check if performance is reasonable
        passed = data['p99_latency_ms'] < 10000  # 10 seconds threshold
        results.add_result("forward_pass_performance", passed, data)

        if passed:
            print("  ✅ Performance acceptable")
        else:
            print("  ❌ Performance too slow")

    except Exception as e:
        results.add_result("forward_pass_performance", False, error=str(e))
        print(f"  ❌ Error: {e}")
        traceback.print_exc()

    return results


def test_stability(n_steps=1000):
    """Test integration stability over long runs."""
    print(f"\n[3/8] Testing integration stability ({n_steps} steps)...")
    results = StressTestResults()

    try:
        import torch
        from grcm.echozero import build_ring_lattice, build_coupling_matrix, integrate_echozero

        n_nodes = 64
        adjacency, node_freqs, edges = build_ring_lattice(n_nodes)
        K = build_coupling_matrix(n_nodes, edges)

        psi0 = torch.complex(
            torch.randn(n_nodes) * 0.1,
            torch.randn(n_nodes) * 0.1,
        )
        I_t = torch.zeros(n_nodes, dtype=torch.complex64)
        desires = torch.zeros(n_nodes)

        # Integrate
        start = time.time()
        psi_final, trajectory, times = integrate_echozero(
            psi0=psi0,
            t_span=(0.0, n_steps * 0.01),
            I_t=I_t,
            desires=desires,
            node_freqs=node_freqs,
            K=K,
            dt=0.01,
            save_trajectory=True,
        )
        elapsed = time.time() - start

        # Check stability
        is_finite = torch.isfinite(psi_final).all().item()
        max_magnitude = psi_final.abs().max().item()

        data = {
            "n_steps": n_steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": n_steps / elapsed,
            "is_finite": is_finite,
            "max_magnitude": max_magnitude,
            "final_mean_amplitude": psi_final.abs().mean().item(),
        }

        print(f"  Integration time: {elapsed:.2f}s")
        print(f"  Steps/sec: {data['steps_per_sec']:.1f}")
        print(f"  Final max |ψ|: {max_magnitude:.4f}")
        print(f"  Finite: {is_finite}")

        passed = is_finite and max_magnitude < 100.0
        results.add_result("stability", passed, data)

        if passed:
            print("  ✅ Stable")
        else:
            print("  ❌ Diverged or NaN")

    except Exception as e:
        results.add_result("stability", False, error=str(e))
        print(f"  ❌ Error: {e}")
        traceback.print_exc()

    return results


def test_training_speed(n_steps=100):
    """Benchmark EchoMirror training speed."""
    print(f"\n[4/8] Testing EchoMirror training speed ({n_steps} steps)...")
    results = StressTestResults()

    try:
        import torch
        from grcm.hybrid import EchoGRCMHybrid
        from grcm.train import EchoMirrorTrainer, MultimodalDatastream

        model = EchoGRCMHybrid(n_nodes=64)
        trainer = EchoMirrorTrainer(learning_rate=0.001)
        datastream = MultimodalDatastream(batch_size=8)

        start = time.time()

        for step in range(n_steps):
            batch = datastream.generate_batch()

            with torch.no_grad():
                outputs = model(**batch)

            K_new, stats = trainer.batch_update(
                K=model.echozero.K,
                psi_batch=outputs["psi"],
                coherence_batch=outputs["coherence"],
            )

            model.echozero.K.copy_(K_new)

        elapsed = time.time() - start

        data = {
            "n_steps": n_steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": n_steps / elapsed,
            "batch_size": 8,
        }

        print(f"  Training time: {elapsed:.2f}s")
        print(f"  Steps/sec: {data['steps_per_sec']:.1f}")
        print(f"  Samples/sec: {data['steps_per_sec'] * 8:.1f}")

        passed = True
        results.add_result("training_speed", passed, data)
        print("  ✅ Training completed")

    except Exception as e:
        results.add_result("training_speed", False, error=str(e))
        print(f"  ❌ Error: {e}")
        traceback.print_exc()

    return results


def test_scalability():
    """Test scalability across different node counts."""
    print("\n[5/8] Testing scalability...")
    results = StressTestResults()

    scales = [16, 32, 64, 128, 256]

    try:
        import torch
        from grcm.hybrid import EchoGRCMHybrid

        scaling_data = []

        for n_nodes in scales:
            print(f"  Testing N={n_nodes}...")

            model = EchoGRCMHybrid(n_nodes=n_nodes)

            image_emb = torch.randn(1, 512)
            audio_emb = torch.randn(1, 768)
            action = torch.zeros(1, 4)

            # Warmup
            with torch.no_grad():
                _ = model(image_emb, audio_emb, action)

            # Benchmark
            n_trials = 10
            start = time.time()

            for _ in range(n_trials):
                with torch.no_grad():
                    outputs = model(image_emb, audio_emb, action)

            elapsed = time.time() - start
            avg_latency = (elapsed / n_trials) * 1000  # ms

            scaling_data.append({
                "n_nodes": n_nodes,
                "latency_ms": avg_latency,
                "phi": outputs["phi"][0].item(),
            })

            print(f"    Latency: {avg_latency:.2f}ms, Φ: {outputs['phi'][0].item():.4f}")

        data = {"scaling": scaling_data}
        results.add_result("scalability", True, data)
        print("  ✅ Scalability test completed")

    except Exception as e:
        results.add_result("scalability", False, error=str(e))
        print(f"  ❌ Error: {e}")
        traceback.print_exc()

    return results


def test_memory_usage():
    """Estimate memory usage."""
    print("\n[6/8] Testing memory usage...")
    results = StressTestResults()

    try:
        import torch
        from grcm.scale import ScalableLatticeBuilder

        builder = ScalableLatticeBuilder()

        scales = [64, 256, 1024, 4096, 16384]
        memory_data = []

        for n_nodes in scales:
            mem = builder.estimate_memory(n_nodes, sparse=(n_nodes > 1000))
            memory_data.append({
                "n_nodes": n_nodes,
                "total_mb": mem["total_mb"],
                "sparse": n_nodes > 1000,
            })

            print(f"  N={n_nodes:6d}: {mem['total_mb']:8.2f} MB (sparse={n_nodes > 1000})")

        data = {"memory": memory_data}
        results.add_result("memory_usage", True, data)
        print("  ✅ Memory estimates computed")

    except Exception as e:
        results.add_result("memory_usage", False, error=str(e))
        print(f"  ❌ Error: {e}")
        traceback.print_exc()

    return results


def test_edge_cases():
    """Test edge cases and failure modes."""
    print("\n[7/8] Testing edge cases...")
    results = StressTestResults()

    try:
        import torch
        from grcm.echozero import compute_coherence, compute_want_modulation

        # Test 1: Zero input
        print("  Testing zero input...")
        psi = torch.zeros(64, dtype=torch.complex64)
        node_freqs = torch.ones(64)
        coherence = compute_coherence(psi, node_freqs)
        assert (coherence >= 0).all() and (coherence <= 1).all()
        print("    ✅ Zero input handled")

        # Test 2: Large magnitude
        print("  Testing large magnitude...")
        psi = torch.complex(
            torch.randn(64) * 10.0,
            torch.randn(64) * 10.0,
        )
        desires = torch.randn(64)
        gamma = compute_want_modulation(psi, desires)
        assert torch.isfinite(gamma).all()
        print("    ✅ Large magnitude handled")

        # Test 3: Boundary frequencies
        print("  Testing boundary frequencies...")
        psi = torch.complex(torch.ones(64), torch.zeros(64))
        node_freqs = torch.ones(64)
        coherence = compute_coherence(psi, node_freqs)
        assert coherence.mean() > 0.5  # Should be highly coherent
        print("    ✅ Boundary frequencies handled")

        results.add_result("edge_cases", True)
        print("  ✅ All edge cases passed")

    except Exception as e:
        results.add_result("edge_cases", False, error=str(e))
        print(f"  ❌ Error: {e}")
        traceback.print_exc()

    return results


def test_hermiticity():
    """Verify hermiticity is maintained."""
    print("\n[8/8] Testing hermiticity preservation...")
    results = StressTestResults()

    try:
        import torch
        from grcm.echozero import build_ring_lattice, build_coupling_matrix
        from grcm.echozero.dynamics import enforce_hermiticity

        n_nodes = 64
        adjacency, node_freqs, edges = build_ring_lattice(n_nodes)
        K = build_coupling_matrix(n_nodes, edges)

        # Check initial hermiticity
        K_conj_T = K.conj().T
        diff_initial = (K - K_conj_T).abs().max().item()

        print(f"  Initial hermiticity error: {diff_initial:.2e}")

        # Perturb and re-enforce
        K_perturbed = K + torch.randn_like(K) * 0.01
        K_restored = enforce_hermiticity(K_perturbed)

        diff_restored = (K_restored - K_restored.conj().T).abs().max().item()

        print(f"  Restored hermiticity error: {diff_restored:.2e}")

        data = {
            "initial_error": diff_initial,
            "restored_error": diff_restored,
        }

        passed = diff_initial < 1e-6 and diff_restored < 1e-6
        results.add_result("hermiticity", passed, data)

        if passed:
            print("  ✅ Hermiticity maintained")
        else:
            print("  ❌ Hermiticity violated")

    except Exception as e:
        results.add_result("hermiticity", False, error=str(e))
        print(f"  ❌ Error: {e}")
        traceback.print_exc()

    return results


def main():
    """Run all stress tests."""
    print("=" * 80)
    print("  EchoZero + GRCM Stress Test Suite")
    print("  Version 1.0")
    print("=" * 80)

    all_results = []

    # Run all tests
    all_results.append(test_imports())
    all_results.append(test_forward_pass_performance(n_trials=100))
    all_results.append(test_stability(n_steps=1000))
    all_results.append(test_training_speed(n_steps=100))
    all_results.append(test_scalability())
    all_results.append(test_memory_usage())
    all_results.append(test_edge_cases())
    all_results.append(test_hermiticity())

    # Aggregate results
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)

    print("\n" + "=" * 80)
    print("  OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_failed} ❌")

    if total_failed > 0:
        print("\n⚠️  Some tests failed. See details above.")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
