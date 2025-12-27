"""
GRCM Benchmark Suite
Performance testing and latency measurement
"""
import torch
import time
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json

from .core import ModularGRCM
from .config import GRCMConfig


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    num_warmup: int = 10
    num_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    measure_memory: bool = True
    measure_phi: bool = True
    target_latency_ms: float = 50.0


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    throughput_samples_per_sec: float
    meets_target: bool


class GRCMBenchmark:
    """
    Comprehensive benchmark suite for GRCM

    Measures:
    - Forward pass latency
    - Throughput
    - Memory usage
    - Phi computation overhead
    - Batch size scaling
    """

    def __init__(
        self,
        model: ModularGRCM,
        config: Optional[BenchmarkConfig] = None,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.config = config or BenchmarkConfig()
        self.device = device
        self.results = {}

    def _prepare_inputs(self, batch_size: int) -> tuple:
        """Prepare dummy inputs for benchmarking"""
        image_emb = torch.randn(batch_size, 512, device=self.device)
        audio_emb = torch.randn(batch_size, 768, device=self.device)
        action = torch.randn(batch_size, 4, device=self.device)
        return image_emb, audio_emb, action

    def _measure_latency(
        self,
        forward_fn: Callable,
        num_iterations: int = 100
    ) -> List[float]:
        """
        Measure latency over multiple iterations

        Args:
            forward_fn: Function to benchmark
            num_iterations: Number of timing runs

        Returns:
            List of latencies in milliseconds
        """
        latencies = []

        for _ in range(num_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            start = time.perf_counter()
            forward_fn()
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        return latencies

    def benchmark_forward_pass(
        self,
        batch_size: int = 1,
        with_grad: bool = False
    ) -> BenchmarkResult:
        """
        Benchmark forward pass latency

        Args:
            batch_size: Batch size to test
            with_grad: Enable gradient computation

        Returns:
            BenchmarkResult with timing statistics
        """
        print(f"\n[Benchmark] Forward pass (batch_size={batch_size}, grad={with_grad})")

        # Prepare inputs
        inputs = self._prepare_inputs(batch_size)

        # Warmup
        print(f"  Warmup: {self.config.num_warmup} iterations...")
        for _ in range(self.config.num_warmup):
            if with_grad:
                _ = self.model(*inputs)
            else:
                with torch.no_grad():
                    _ = self.model(*inputs)

        # Benchmark
        print(f"  Measuring: {self.config.num_iterations} iterations...")

        def forward_fn():
            if with_grad:
                return self.model(*inputs)
            else:
                with torch.no_grad():
                    return self.model(*inputs)

        latencies = self._measure_latency(forward_fn, self.config.num_iterations)

        # Compute statistics
        mean_ms = statistics.mean(latencies)
        std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        min_ms = min(latencies)
        max_ms = max(latencies)
        median_ms = statistics.median(latencies)

        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p95_ms = sorted_latencies[p95_idx]
        p99_ms = sorted_latencies[p99_idx]

        # Throughput (samples per second)
        throughput = (batch_size * 1000) / mean_ms

        # Check if meets target
        meets_target = mean_ms < self.config.target_latency_ms

        result = BenchmarkResult(
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            median_ms=median_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            throughput_samples_per_sec=throughput,
            meets_target=meets_target
        )

        # Print results
        status = "✓" if meets_target else "✗"
        print(f"  {status} Mean: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        print(f"    Median: {median_ms:.2f}ms | P95: {p95_ms:.2f}ms | P99: {p99_ms:.2f}ms")
        print(f"    Throughput: {throughput:.1f} samples/sec")
        print(f"    Target: {self.config.target_latency_ms}ms - {'MET' if meets_target else 'MISSED'}")

        return result

    def benchmark_batch_scaling(self) -> Dict[int, BenchmarkResult]:
        """
        Benchmark performance across different batch sizes

        Returns:
            Dictionary mapping batch_size -> BenchmarkResult
        """
        print("\n" + "=" * 60)
        print("Batch Size Scaling Benchmark")
        print("=" * 60)

        results = {}

        for batch_size in self.config.batch_sizes:
            result = self.benchmark_forward_pass(batch_size, with_grad=False)
            results[batch_size] = result

        self.results['batch_scaling'] = results
        return results

    def benchmark_phi_overhead(self) -> Dict[str, Any]:
        """
        Measure overhead of Phi computation

        Returns:
            Overhead analysis
        """
        print("\n" + "=" * 60)
        print("Phi Computation Overhead")
        print("=" * 60)

        batch_size = 4
        inputs = self._prepare_inputs(batch_size)

        # Time full forward pass
        def full_forward():
            with torch.no_grad():
                return self.model(*inputs)

        full_latencies = self._measure_latency(full_forward, 50)
        full_mean = statistics.mean(full_latencies)

        print(f"  Full forward: {full_mean:.2f}ms")

        # Phi is computed at the end, so overhead is minimal
        # This is mainly for analysis
        result = {
            'full_forward_ms': full_mean,
            'phi_overhead_estimate_ms': full_mean * 0.05,  # Estimate ~5%
            'note': 'Phi is lightweight (no backprop overhead)'
        }

        self.results['phi_overhead'] = result
        return result

    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """
        Measure memory usage

        Returns:
            Memory statistics
        """
        print("\n" + "=" * 60)
        print("Memory Usage")
        print("=" * 60)

        if not self.config.measure_memory:
            return {}

        batch_size = 4
        inputs = self._prepare_inputs(batch_size)

        # Measure parameter memory
        param_memory = sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        ) / (1024 ** 2)

        # Measure activation memory (rough estimate)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        with torch.no_grad():
            _ = self.model(*inputs)

        result = {
            'param_memory_mb': param_memory,
            'estimated_total_mb': param_memory * 2,  # Params + activations estimate
            'device': self.device
        }

        print(f"  Parameter memory: {param_memory:.2f} MB")
        print(f"  Estimated total: {result['estimated_total_mb']:.2f} MB")

        self.results['memory'] = result
        return result

    def benchmark_coherence_distribution(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze coherence score distribution

        Args:
            num_samples: Number of forward passes to analyze

        Returns:
            Coherence statistics
        """
        print("\n" + "=" * 60)
        print("Coherence Distribution Analysis")
        print("=" * 60)

        coherence_scores = []
        batch_size = 4

        with torch.no_grad():
            for _ in range(num_samples):
                inputs = self._prepare_inputs(batch_size)
                outputs = self.model(*inputs)
                coherence_scores.append(outputs['coherence'].mean().item())

        mean_coh = statistics.mean(coherence_scores)
        std_coh = statistics.stdev(coherence_scores)
        above_threshold = sum(1 for c in coherence_scores if c > 0.7) / len(coherence_scores)

        result = {
            'mean_coherence': mean_coh,
            'std_coherence': std_coh,
            'min_coherence': min(coherence_scores),
            'max_coherence': max(coherence_scores),
            'above_threshold_ratio': above_threshold,
            'threshold': 0.7
        }

        print(f"  Mean coherence: {mean_coh:.3f} ± {std_coh:.3f}")
        print(f"  Above threshold (0.7): {above_threshold:.1%}")

        self.results['coherence'] = result
        return result

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark suite

        Returns:
            All benchmark results
        """
        print("\n" + "=" * 60)
        print("GRCM FULL BENCHMARK SUITE")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Target latency: {self.config.target_latency_ms}ms")

        # Run all benchmarks
        self.benchmark_batch_scaling()
        self.benchmark_phi_overhead()
        self.benchmark_memory_usage()
        self.benchmark_coherence_distribution()

        # Summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        if 'batch_scaling' in self.results:
            print("\nLatency by Batch Size:")
            for bs, result in self.results['batch_scaling'].items():
                status = "✓" if result.meets_target else "✗"
                print(f"  {status} Batch {bs:2d}: {result.mean_ms:6.2f}ms (throughput: {result.throughput_samples_per_sec:6.1f} samples/sec)")

        if 'memory' in self.results:
            mem = self.results['memory']
            print(f"\nMemory: {mem['param_memory_mb']:.2f} MB (params)")

        if 'coherence' in self.results:
            coh = self.results['coherence']
            print(f"\nCoherence: {coh['mean_coherence']:.3f} ({coh['above_threshold_ratio']:.1%} above threshold)")

        print("\n" + "=" * 60)

    def export_results(self, output_path: str):
        """Export results to JSON"""
        # Convert BenchmarkResult objects to dicts
        export_data = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                export_data[key] = {}
                for k, v in value.items():
                    if isinstance(v, BenchmarkResult):
                        export_data[key][k] = v.__dict__
                    else:
                        export_data[key][k] = v
            else:
                export_data[key] = value

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\n[Export] Results saved to {output_path}")


def quick_benchmark(
    config_path: str = "config/grcm_default.yaml",
    device: str = "cpu",
    target_latency_ms: float = 50.0
) -> Dict[str, Any]:
    """
    Quick benchmark convenience function

    Args:
        config_path: Path to GRCM config
        device: Device to run on
        target_latency_ms: Target latency threshold

    Returns:
        Benchmark results
    """
    from .core import ModularGRCM
    from .config import load_config

    # Load model
    config = load_config(config_path)
    model = ModularGRCM(config)

    # Create benchmark
    bench_config = BenchmarkConfig(target_latency_ms=target_latency_ms)
    benchmark = GRCMBenchmark(model, bench_config, device)

    # Run
    results = benchmark.run_full_benchmark()

    return results
