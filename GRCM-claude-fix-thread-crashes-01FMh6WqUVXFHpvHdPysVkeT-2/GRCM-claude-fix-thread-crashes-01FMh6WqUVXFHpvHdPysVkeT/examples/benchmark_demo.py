"""
GRCM Benchmark Demo
Comprehensive performance testing and analysis
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm import ModularGRCM, load_config, GRCMBenchmark, BenchmarkConfig


def main():
    print("=" * 70)
    print("GRCM Benchmark Demo")
    print("=" * 70)

    # 1. Load model
    print("\n[1] Loading GRCM model...")
    config = load_config("config/grcm_default.yaml")
    model = ModularGRCM(config)
    model.eval()
    print("    ✓ Model loaded")

    # 2. Configure benchmark
    print("\n[2] Configuring benchmark...")
    bench_config = BenchmarkConfig(
        num_warmup=10,
        num_iterations=100,
        batch_sizes=[1, 2, 4, 8],
        measure_memory=True,
        measure_phi=True,
        target_latency_ms=50.0
    )
    print(f"    Warmup iterations: {bench_config.num_warmup}")
    print(f"    Benchmark iterations: {bench_config.num_iterations}")
    print(f"    Batch sizes: {bench_config.batch_sizes}")
    print(f"    Target latency: {bench_config.target_latency_ms}ms")

    # 3. Create benchmark
    print("\n[3] Creating GRCMBenchmark...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")

    benchmark = GRCMBenchmark(model, bench_config, device)
    print("    ✓ Benchmark ready")

    # 4. Run individual benchmarks
    print("\n[4] Running individual benchmarks...")

    # 4a. Single batch benchmark
    print("\n  [4a] Single batch forward pass...")
    result_bs1 = benchmark.benchmark_forward_pass(batch_size=1, with_grad=False)

    # 4b. Batch size 4 benchmark
    print("\n  [4b] Batch size 4 forward pass...")
    result_bs4 = benchmark.benchmark_forward_pass(batch_size=4, with_grad=False)

    # 5. Batch scaling analysis
    print("\n[5] Batch size scaling analysis...")
    batch_results = benchmark.benchmark_batch_scaling()

    # 6. Phi overhead analysis
    print("\n[6] Phi computation overhead...")
    phi_overhead = benchmark.benchmark_phi_overhead()

    # 7. Memory usage
    print("\n[7] Memory usage analysis...")
    memory_stats = benchmark.benchmark_memory_usage()

    # 8. Coherence distribution
    print("\n[8] Coherence distribution...")
    coherence_stats = benchmark.benchmark_coherence_distribution(num_samples=50)

    # 9. Print detailed analysis
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)

    print("\n[Latency Analysis]")
    print(f"  Batch size 1:")
    print(f"    Mean: {result_bs1.mean_ms:.2f}ms ± {result_bs1.std_ms:.2f}ms")
    print(f"    Min/Max: {result_bs1.min_ms:.2f}ms / {result_bs1.max_ms:.2f}ms")
    print(f"    P95/P99: {result_bs1.p95_ms:.2f}ms / {result_bs1.p99_ms:.2f}ms")
    print(f"    Throughput: {result_bs1.throughput_samples_per_sec:.1f} samples/sec")

    print(f"\n  Batch size 4:")
    print(f"    Mean: {result_bs4.mean_ms:.2f}ms ± {result_bs4.std_ms:.2f}ms")
    print(f"    Throughput: {result_bs4.throughput_samples_per_sec:.1f} samples/sec")
    print(f"    Efficiency: {(result_bs4.throughput_samples_per_sec / result_bs1.throughput_samples_per_sec):.2f}x")

    print("\n[Target Compliance]")
    total_batches = len(batch_results)
    met_target = sum(1 for r in batch_results.values() if r.meets_target)
    compliance_rate = met_target / total_batches
    print(f"  Target latency: {bench_config.target_latency_ms}ms")
    print(f"  Batches meeting target: {met_target}/{total_batches} ({compliance_rate:.1%})")

    if compliance_rate >= 0.75:
        print(f"  ✓ PASS: {compliance_rate:.1%} compliance")
    else:
        print(f"  ⚠  WARNING: Only {compliance_rate:.1%} compliance")

    print("\n[Coherence Analysis]")
    print(f"  Mean coherence: {coherence_stats['mean_coherence']:.3f}")
    print(f"  Std deviation: {coherence_stats['std_coherence']:.3f}")
    print(f"  Above threshold (0.7): {coherence_stats['above_threshold_ratio']:.1%}")

    if coherence_stats['above_threshold_ratio'] >= 0.95:
        print(f"  ✓ PASS: {coherence_stats['above_threshold_ratio']:.1%} above threshold")
    else:
        print(f"  ⚠  WARNING: Only {coherence_stats['above_threshold_ratio']:.1%} above threshold")

    print("\n[Memory Usage]")
    print(f"  Parameter memory: {memory_stats['param_memory_mb']:.2f} MB")
    print(f"  Estimated total: {memory_stats['estimated_total_mb']:.2f} MB")

    if memory_stats['estimated_total_mb'] < 500:
        print(f"  ✓ PASS: Memory usage under 500 MB")
    else:
        print(f"  ⚠  WARNING: Memory usage exceeds 500 MB")

    # 10. Export results
    print("\n[10] Exporting results...")
    Path("benchmark_results").mkdir(exist_ok=True)
    benchmark.export_results("benchmark_results/grcm_benchmark.json")
    print("    ✓ Results exported to benchmark_results/grcm_benchmark.json")

    # 11. Final summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    all_pass = True

    # Check 1: Latency target
    if compliance_rate >= 0.75:
        print("  ✓ Latency: PASS")
    else:
        print("  ✗ Latency: FAIL")
        all_pass = False

    # Check 2: Coherence
    if coherence_stats['above_threshold_ratio'] >= 0.95:
        print("  ✓ Coherence: PASS")
    else:
        print("  ✗ Coherence: FAIL")
        all_pass = False

    # Check 3: Memory
    if memory_stats['estimated_total_mb'] < 500:
        print("  ✓ Memory: PASS")
    else:
        print("  ✗ Memory: FAIL")
        all_pass = False

    # Check 4: Phi stability
    phi_stable = coherence_stats['std_coherence'] < 0.2
    if phi_stable:
        print("  ✓ Phi Stability: PASS")
    else:
        print("  ✗ Phi Stability: FAIL")
        all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("✓ ALL BENCHMARKS PASSED!")
    else:
        print("⚠  SOME BENCHMARKS FAILED - Review results above")
    print("=" * 70)

    # Recommendations
    print("\nRecommendations:")
    if result_bs1.mean_ms > 50:
        print("  - Consider quantization to reduce latency")
    if memory_stats['estimated_total_mb'] > 300:
        print("  - Consider pruning or distillation for memory reduction")
    if coherence_stats['above_threshold_ratio'] < 0.95:
        print("  - Review bandwidth and coherence threshold settings")

    print("\nNext steps:")
    print("  - Apply optimizations: python examples/optimization_demo.py")
    print("  - Review benchmark JSON for detailed metrics")
    print("  - Test with production workload patterns")


if __name__ == "__main__":
    main()
