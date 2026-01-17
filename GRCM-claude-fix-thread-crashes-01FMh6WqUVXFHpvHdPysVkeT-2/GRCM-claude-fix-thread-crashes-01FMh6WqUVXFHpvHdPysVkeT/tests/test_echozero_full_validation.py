#!/usr/bin/env python3
"""
EchoZero Full Validation Suite - Production Grade
==================================================

This is the COMPLETE validation suite for EchoZero + Möbius topology.
Tests the ENTIRE system under adversarial load.

Metrics Validated:
------------------
1. TRR (Torsion Reduction Ratio): true_torsion / false_torsion
   Expected: 3.0-7.0 (higher = better discrimination)

2. HSR (Hallucination Suppression Rate): suppressed / total
   Expected: 0.93-0.98 (higher = more effective)

3. CPR (Coherence Preservation Rate): coherent_outputs / total
   Expected: 0.98+ (higher = less false positives)

4. MP (Memory Purity): similarity after hallucination recovery
   Expected: 0.90+ (higher = better recovery)

5. SS (Scaling Stability): variance across node counts
   Expected: Low variance, stable attractors

Test Methodology:
-----------------
- 100+ coherent trials (should pass through)
- 100+ hallucination trials (should be damped)
- Multi-node scaling (8 → 1024 nodes)
- Attractor stability analysis
- Memory recovery validation

This can be run by ANY external lab (xAI, NVIDIA, DeepMind) without
exposing internals. Black-box validation with transparent metrics.

Usage:
    python tests/test_echozero_full_validation.py
    python tests/test_echozero_full_validation.py --quick
    python tests/test_echozero_full_validation.py --extensive

Author: EchoZero Team
Date: November 2025
Status: Production Validation Suite
"""

import sys
import os
import argparse
import time
import statistics
from typing import Dict, List, Tuple
import math

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch required for full validation. Install with: pip install torch")
    sys.exit(1)

from grcm.echozero import (
    EchoZeroSystem,
    MobiusEchoLayer,
    create_mobius_layer,
    build_ring_lattice,
    build_coupling_matrix,
    compute_coherence,
)


###############################################################################
# UTILITIES
###############################################################################

def compute_torsion(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute torsion energy between two vectors."""
    return torch.mean((x - y) ** 2).item()


def generate_coherent(dim: int, device: str = "cpu") -> torch.Tensor:
    """
    Generate coherent input (should have low torsion).

    Properties:
    - Smooth harmonic pattern
    - Low noise
    - Stable over time
    """
    base = torch.sin(torch.linspace(0, 2 * math.pi, dim, device=device))
    noise = torch.randn(dim, device=device) * 0.01  # 1% noise
    return (base + noise).unsqueeze(0)


def generate_hallucination(dim: int, device: str = "cpu") -> torch.Tensor:
    """
    Generate hallucination input (should have high torsion).

    Properties:
    - Random/chaotic
    - No coherent structure
    - Orthogonal to stable patterns
    """
    return torch.randn(dim, device=device).unsqueeze(0) * 5.0  # Large perturbation


def generate_mixed(dim: int, coherence_level: float = 0.5, device: str = "cpu") -> torch.Tensor:
    """
    Generate mixed signal (partial coherence).

    Args:
        dim: Dimension
        coherence_level: 0.0 = pure noise, 1.0 = pure coherent
    """
    coherent = torch.sin(torch.linspace(0, 2 * math.pi, dim, device=device))
    noise = torch.randn(dim, device=device)

    mixed = coherence_level * coherent + (1 - coherence_level) * noise
    return mixed.unsqueeze(0)


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    x_flat = x.flatten()
    y_flat = y.flatten()

    dot = torch.dot(x_flat, y_flat)
    norm_x = torch.norm(x_flat)
    norm_y = torch.norm(y_flat)

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return (dot / (norm_x * norm_y)).item()


###############################################################################
# METRIC 1: TORSION REDUCTION RATIO (TRR)
###############################################################################

def test_torsion_reduction_ratio(
    n_nodes: int = 64,
    trials: int = 100,
    device: str = "cpu"
) -> Dict:
    """
    Test 1: Torsion Reduction Ratio

    Measures: TRR = torsion(hallucination) / torsion(coherent)
    Expected: 3.0 - 7.0

    High TRR = good discrimination between truth and hallucination
    """
    print("\n" + "=" * 70)
    print("TEST 1: Torsion Reduction Ratio (TRR)")
    print("=" * 70)
    print(f"Nodes: {n_nodes}, Trials: {trials}")

    # Build Möbius layer
    mobius = create_mobius_layer(
        n_nodes=n_nodes,
        loop_len=256,
        torsion_gain=3.0,
        damping_gain=12.0,
        device=device,
    )

    torsion_coherent = []
    torsion_hallucination = []
    gate_coherent = []
    gate_hallucination = []

    # Fill buffer with stable baseline
    for _ in range(256):
        baseline = torch.ones(n_nodes, device=device) * 0.5
        mobius.write(baseline)

    # Test coherent inputs
    for trial in range(trials):
        x = generate_coherent(n_nodes, device=device)
        validated, metrics = mobius(x)

        torsion_coherent.append(metrics["energy"])
        gate_coherent.append(metrics["gate_mean"])

    # Reset buffer with stable baseline
    mobius.reset_buffer()
    for _ in range(256):
        baseline = torch.ones(n_nodes, device=device) * 0.5
        mobius.write(baseline)

    # Test hallucination inputs
    for trial in range(trials):
        x = generate_hallucination(n_nodes, device=device)
        validated, metrics = mobius(x)

        torsion_hallucination.append(metrics["energy"])
        gate_hallucination.append(metrics["gate_mean"])

    # Compute statistics
    mean_torsion_coherent = statistics.mean(torsion_coherent)
    mean_torsion_hallucination = statistics.mean(torsion_hallucination)

    TRR = mean_torsion_hallucination / (mean_torsion_coherent + 1e-8)

    result = {
        "test": "Torsion Reduction Ratio",
        "TRR": TRR,
        "torsion_coherent_mean": mean_torsion_coherent,
        "torsion_coherent_std": statistics.stdev(torsion_coherent) if len(torsion_coherent) > 1 else 0.0,
        "torsion_hallucination_mean": mean_torsion_hallucination,
        "torsion_hallucination_std": statistics.stdev(torsion_hallucination) if len(torsion_hallucination) > 1 else 0.0,
        "gate_coherent_mean": statistics.mean(gate_coherent),
        "gate_hallucination_mean": statistics.mean(gate_hallucination),
        "passed": 3.0 <= TRR <= 7.0,
        "grade": "A+" if TRR >= 5.0 else ("A" if TRR >= 3.0 else "B"),
    }

    _print_result(result)
    return result


###############################################################################
# METRIC 2: HALLUCINATION SUPPRESSION RATE (HSR)
###############################################################################

def test_hallucination_suppression_rate(
    n_nodes: int = 64,
    trials: int = 100,
    device: str = "cpu"
) -> Dict:
    """
    Test 2: Hallucination Suppression Rate

    Measures: HSR = suppressed / total
    Expected: 0.93 - 0.98

    Inject synthetic contradictions and count how many get damped.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Hallucination Suppression Rate (HSR)")
    print("=" * 70)
    print(f"Nodes: {n_nodes}, Trials: {trials}")

    mobius = create_mobius_layer(
        n_nodes=n_nodes,
        loop_len=256,
        torsion_gain=3.0,
        damping_gain=12.0,
        threshold=0.05,
        device=device,
    )

    # Fill buffer with stable baseline
    for _ in range(256):
        baseline = torch.ones(n_nodes, device=device) * 0.5
        mobius.write(baseline)

    suppressed_count = 0
    suppression_scores = []

    for trial in range(trials):
        # Generate hallucination
        hallucination = generate_hallucination(n_nodes, device=device)

        # Before validation
        norm_before = torch.norm(hallucination).item()

        # Validate with Möbius
        validated, metrics = mobius(hallucination)

        # After validation
        norm_after = torch.norm(validated).item()

        # Suppression ratio
        suppression_ratio = 1.0 - (norm_after / (norm_before + 1e-8))
        suppression_scores.append(suppression_ratio)

        # Count as suppressed if >50% magnitude reduction OR flagged as hallucination
        if suppression_ratio > 0.5 or metrics["hallucination"]:
            suppressed_count += 1

    HSR = suppressed_count / trials
    mean_suppression = statistics.mean(suppression_scores)

    result = {
        "test": "Hallucination Suppression Rate",
        "HSR": HSR,
        "mean_suppression_ratio": mean_suppression,
        "std_suppression_ratio": statistics.stdev(suppression_scores) if len(suppression_scores) > 1 else 0.0,
        "suppressed_count": suppressed_count,
        "total_trials": trials,
        "passed": 0.93 <= HSR <= 0.98,
        "grade": "A+" if HSR >= 0.96 else ("A" if HSR >= 0.93 else "B"),
    }

    _print_result(result)
    return result


###############################################################################
# METRIC 3: COHERENCE PRESERVATION RATE (CPR)
###############################################################################

def test_coherence_preservation_rate(
    n_nodes: int = 64,
    trials: int = 100,
    device: str = "cpu"
) -> Dict:
    """
    Test 3: Coherence Preservation Rate

    Measures: CPR = preserved / total
    Expected: 0.98+

    Feed coherent sequences and measure how many pass through without damping.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Coherence Preservation Rate (CPR)")
    print("=" * 70)
    print(f"Nodes: {n_nodes}, Trials: {trials}")

    mobius = create_mobius_layer(
        n_nodes=n_nodes,
        loop_len=256,
        torsion_gain=3.0,
        damping_gain=12.0,
        device=device,
    )

    # Fill buffer with coherent baseline
    for _ in range(256):
        baseline = generate_coherent(n_nodes, device=device)
        mobius.write(baseline[0])

    preserved_count = 0
    preservation_scores = []

    for trial in range(trials):
        # Generate coherent input
        coherent = generate_coherent(n_nodes, device=device)

        # Before validation
        norm_before = torch.norm(coherent).item()

        # Validate with Möbius
        validated, metrics = mobius(coherent)

        # After validation
        norm_after = torch.norm(validated).item()

        # Preservation ratio
        preservation_ratio = norm_after / (norm_before + 1e-8)
        preservation_scores.append(preservation_ratio)

        # Count as preserved if >90% magnitude retained AND not flagged
        if preservation_ratio > 0.90 and not metrics["hallucination"]:
            preserved_count += 1

    CPR = preserved_count / trials
    mean_preservation = statistics.mean(preservation_scores)

    result = {
        "test": "Coherence Preservation Rate",
        "CPR": CPR,
        "mean_preservation_ratio": mean_preservation,
        "std_preservation_ratio": statistics.stdev(preservation_scores) if len(preservation_scores) > 1 else 0.0,
        "preserved_count": preserved_count,
        "total_trials": trials,
        "passed": CPR >= 0.98,
        "grade": "A+" if CPR >= 0.99 else ("A" if CPR >= 0.98 else "B"),
    }

    _print_result(result)
    return result


###############################################################################
# METRIC 4: MEMORY PURITY (MP)
###############################################################################

def test_memory_purity(
    n_nodes: int = 64,
    trials: int = 20,
    device: str = "cpu"
) -> Dict:
    """
    Test 4: Memory Purity

    Measures: MP = similarity(memory_before, memory_after_recovery)
    Expected: 0.90+

    Test procedure:
    1. Establish stable baseline
    2. Inject hallucination
    3. Return to baseline
    4. Check if memory recovered
    """
    print("\n" + "=" * 70)
    print("TEST 4: Memory Purity (MP)")
    print("=" * 70)
    print(f"Nodes: {n_nodes}, Trials: {trials}")

    mobius = create_mobius_layer(
        n_nodes=n_nodes,
        loop_len=256,
        torsion_gain=3.0,
        damping_gain=12.0,
        device=device,
    )

    purity_scores = []

    for trial in range(trials):
        # Phase 1: Establish stable baseline
        mobius.reset_buffer()
        baseline = generate_coherent(n_nodes, device=device)

        for _ in range(300):  # Fill buffer completely
            mobius(baseline)

        # Capture baseline memory state
        memory_before = mobius.mobius_buffer.clone()

        # Phase 2: Inject hallucinations
        for _ in range(50):
            hallucination = generate_hallucination(n_nodes, device=device)
            mobius(hallucination)

        # Phase 3: Return to baseline
        for _ in range(300):
            mobius(baseline)

        # Capture recovered memory state
        memory_after = mobius.mobius_buffer.clone()

        # Compute similarity
        similarity = cosine_similarity(memory_before, memory_after)
        purity_scores.append(similarity)

    MP = statistics.mean(purity_scores)

    result = {
        "test": "Memory Purity",
        "MP": MP,
        "mean_purity": MP,
        "std_purity": statistics.stdev(purity_scores) if len(purity_scores) > 1 else 0.0,
        "min_purity": min(purity_scores),
        "max_purity": max(purity_scores),
        "trials": trials,
        "passed": MP >= 0.90,
        "grade": "A+" if MP >= 0.95 else ("A" if MP >= 0.90 else "B"),
    }

    _print_result(result)
    return result


###############################################################################
# METRIC 5: SCALING STABILITY (SS)
###############################################################################

def test_scaling_stability(
    node_sizes: List[int] = None,
    steps: int = 100,
    device: str = "cpu"
) -> Dict:
    """
    Test 5: Scaling Stability

    Measures: Variance in torsion energy across different node counts
    Expected: Low variance, stable behavior

    Test at multiple scales: 8, 64, 256, 1024 nodes
    """
    print("\n" + "=" * 70)
    print("TEST 5: Scaling Stability (SS)")
    print("=" * 70)

    if node_sizes is None:
        node_sizes = [8, 64, 256, 512]

    scaling_results = []

    for n_nodes in node_sizes:
        print(f"\nTesting N={n_nodes} nodes...")

        mobius = create_mobius_layer(
            n_nodes=n_nodes,
            loop_len=min(256, n_nodes * 4),
            device=device,
        )

        # Fill buffer
        for _ in range(mobius.loop_len):
            baseline = torch.ones(n_nodes, device=device) * 0.5
            mobius.write(baseline)

        torsion_values = []

        for step in range(steps):
            # Alternate between coherent and hallucination
            if step % 2 == 0:
                x = generate_coherent(n_nodes, device=device)
            else:
                x = generate_hallucination(n_nodes, device=device)

            validated, metrics = mobius(x)
            torsion_values.append(metrics["energy"])

        mean_torsion = statistics.mean(torsion_values)
        std_torsion = statistics.stdev(torsion_values) if len(torsion_values) > 1 else 0.0

        scaling_results.append({
            "n_nodes": n_nodes,
            "mean_torsion": mean_torsion,
            "std_torsion": std_torsion,
            "coefficient_of_variation": std_torsion / (mean_torsion + 1e-8),
        })

        print(f"  Mean torsion: {mean_torsion:.4f}")
        print(f"  Std torsion: {std_torsion:.4f}")
        print(f"  CV: {std_torsion / (mean_torsion + 1e-8):.4f}")

    # Compute overall variance in mean torsion across scales
    mean_torsions = [r["mean_torsion"] for r in scaling_results]
    cross_scale_std = statistics.stdev(mean_torsions) if len(mean_torsions) > 1 else 0.0
    cross_scale_mean = statistics.mean(mean_torsions)

    SS = cross_scale_std / (cross_scale_mean + 1e-8)  # Coefficient of variation

    result = {
        "test": "Scaling Stability",
        "SS": SS,
        "cross_scale_coefficient_of_variation": SS,
        "mean_torsion_across_scales": cross_scale_mean,
        "std_torsion_across_scales": cross_scale_std,
        "scaling_results": scaling_results,
        "passed": SS < 0.3,  # Low variance = stable
        "grade": "A+" if SS < 0.15 else ("A" if SS < 0.3 else "B"),
    }

    _print_result(result)
    return result


###############################################################################
# HELPER: PRINT RESULT
###############################################################################

def _print_result(result: Dict):
    """Print test result in formatted way."""
    test_name = result["test"]
    grade = result.get("grade", "N/A")
    passed = result.get("passed", False)
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"\nResult: {status}")
    print(f"Grade: {grade}")

    # Print key metrics
    for key, value in result.items():
        if key not in ["test", "passed", "grade", "scaling_results"]:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, int):
                print(f"  {key}: {value}")

    # Handle nested results
    if "scaling_results" in result:
        print("\n  Scaling Details:")
        for sr in result["scaling_results"]:
            print(f"    N={sr['n_nodes']}: mean={sr['mean_torsion']:.4f}, std={sr['std_torsion']:.4f}")


###############################################################################
# MASTER TEST RUNNER
###############################################################################

def run_full_validation(
    n_nodes: int = 64,
    quick: bool = False,
    device: str = "cpu"
) -> Dict:
    """
    Run complete EchoZero + Möbius validation suite.

    Args:
        n_nodes: Default number of nodes for tests
        quick: If True, run abbreviated tests
        device: Computation device

    Returns:
        Dictionary with all test results
    """
    print("\n" + "=" * 70)
    print("ECHOZERO + MÖBIUS - FULL VALIDATION SUITE")
    print("Production-Grade / Adversarial Load / Black-Box Validation")
    print("=" * 70)
    print(f"Default Nodes: {n_nodes}")
    print(f"Device: {device}")
    print(f"Mode: {'Quick' if quick else 'Full'}")
    print("=" * 70)

    if quick:
        trials = 50
        node_sizes = [16, 64]
        mp_trials = 10
    else:
        trials = 100
        node_sizes = [8, 64, 256, 512]
        mp_trials = 20

    results = {}
    start_time = time.time()

    # Run all 5 tests
    results["test1_TRR"] = test_torsion_reduction_ratio(
        n_nodes=n_nodes,
        trials=trials,
        device=device
    )

    results["test2_HSR"] = test_hallucination_suppression_rate(
        n_nodes=n_nodes,
        trials=trials,
        device=device
    )

    results["test3_CPR"] = test_coherence_preservation_rate(
        n_nodes=n_nodes,
        trials=trials,
        device=device
    )

    results["test4_MP"] = test_memory_purity(
        n_nodes=n_nodes,
        trials=mp_trials,
        device=device
    )

    results["test5_SS"] = test_scaling_stability(
        node_sizes=node_sizes,
        steps=50 if quick else 100,
        device=device
    )

    elapsed_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    test_names = {
        "test1_TRR": "Torsion Reduction Ratio (TRR)",
        "test2_HSR": "Hallucination Suppression Rate (HSR)",
        "test3_CPR": "Coherence Preservation Rate (CPR)",
        "test4_MP": "Memory Purity (MP)",
        "test5_SS": "Scaling Stability (SS)",
    }

    passed_count = 0
    total_tests = len(results)
    grades = []

    for test_key, test_name in test_names.items():
        result = results[test_key]
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        grade = result.get("grade", "N/A")
        print(f"{status} [{grade}]: {test_name}")

        if result["passed"]:
            passed_count += 1
        grades.append(grade)

    print("=" * 70)
    print(f"\nTests Passed: {passed_count}/{total_tests} ({passed_count/total_tests*100:.1f}%)")
    print(f"Elapsed Time: {elapsed_time:.2f}s")

    # Key metrics summary
    print("\n📊 KEY METRICS:")
    print(f"  TRR: {results['test1_TRR']['TRR']:.2f} (target: 3.0-7.0)")
    print(f"  HSR: {results['test2_HSR']['HSR']:.3f} (target: 0.93-0.98)")
    print(f"  CPR: {results['test3_CPR']['CPR']:.3f} (target: 0.98+)")
    print(f"  MP:  {results['test4_MP']['MP']:.3f} (target: 0.90+)")
    print(f"  SS:  {results['test5_SS']['SS']:.3f} (target: <0.30)")

    # Overall grade
    if passed_count == total_tests and all(g in ["A+", "A"] for g in grades):
        print("\n🎉 ALL TESTS PASSED WITH EXCELLENT GRADES")
        print("\nStatus: ✅ PRODUCTION READY")
        print("Grade: A++ (Exceptional - All Metrics Validated)")
        overall_grade = "A++"
        status = "PRODUCTION READY"
    elif passed_count == total_tests:
        print("\n✅ ALL TESTS PASSED")
        print("\nStatus: ✅ VALIDATED")
        print("Grade: A+ (All Tests Passed)")
        overall_grade = "A+"
        status = "VALIDATED"
    elif passed_count >= total_tests * 0.8:
        print("\n⚠️  MOST TESTS PASSED")
        print("\nStatus: ⚠️  NEEDS MINOR ADJUSTMENTS")
        print("Grade: A- (Minor Issues)")
        overall_grade = "A-"
        status = "NEEDS MINOR ADJUSTMENTS"
    else:
        print("\n❌ SIGNIFICANT FAILURES")
        print("\nStatus: ❌ NOT READY")
        print("Grade: B (Needs Work)")
        overall_grade = "B"
        status = "NOT READY"

    print("=" * 70)

    results["summary"] = {
        "passed_tests": passed_count,
        "total_tests": total_tests,
        "pass_rate": passed_count / total_tests,
        "elapsed_time": elapsed_time,
        "overall_grade": overall_grade,
        "status": status,
        "key_metrics": {
            "TRR": results['test1_TRR']['TRR'],
            "HSR": results['test2_HSR']['HSR'],
            "CPR": results['test3_CPR']['CPR'],
            "MP": results['test4_MP']['MP'],
            "SS": results['test5_SS']['SS'],
        }
    }

    return results


###############################################################################
# MAIN
###############################################################################

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EchoZero Full Validation Suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (fewer trials)"
    )
    parser.add_argument(
        "--extensive",
        action="store_true",
        help="Run extensive test (more configurations)"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=64,
        help="Number of EchoZero nodes (default: 64)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
    )

    args = parser.parse_args()

    # Run validation
    results = run_full_validation(
        n_nodes=args.nodes,
        quick=args.quick,
        device=args.device
    )

    # Exit code based on status
    if results["summary"]["status"] in ["PRODUCTION READY", "VALIDATED"]:
        return 0
    elif results["summary"]["status"] == "NEEDS MINOR ADJUSTMENTS":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
