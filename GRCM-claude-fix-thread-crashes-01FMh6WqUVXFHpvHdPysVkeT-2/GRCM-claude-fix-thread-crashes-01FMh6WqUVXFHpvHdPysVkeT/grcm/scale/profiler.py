"""
Profiling and benchmarking for EchoZero systems.
"""

import torch
import time
from typing import Dict
import numpy as np

from ..echozero import EchoZeroSystem, integrate_echozero


def profile_system(
    system: EchoZeroSystem,
    n_steps: int = 100,
    batch_size: int = 1,
) -> Dict[str, any]:
    """
    Profile EchoZero system performance.

    Args:
        system: EchoZero system
        n_steps: Number of integration steps
        batch_size: Batch size

    Returns:
        stats: Performance statistics
    """
    device = system.device
    n_nodes = system.n_nodes

    # Initialize state
    psi = torch.complex(
        torch.randn(batch_size, n_nodes, device=device) * 0.1,
        torch.randn(batch_size, n_nodes, device=device) * 0.1,
    )

    I_t = torch.complex(
        torch.randn(batch_size, n_nodes, device=device),
        torch.zeros(batch_size, n_nodes, device=device),
    )

    desires = torch.randn(batch_size, n_nodes, device=device)

    # Warm-up
    for _ in range(10):
        _ = system(psi[0], I_t[0], desires[0])

    # Benchmark
    start = time.time()

    for i in range(batch_size):
        psi_final, _, _ = integrate_echozero(
            psi0=psi[i],
            t_span=(0.0, n_steps * 0.01),
            I_t=I_t[i],
            desires=desires[i],
            node_freqs=system.node_freqs,
            K=system.K,
            device=device,
        )

    elapsed = time.time() - start

    # Statistics
    stats = {
        "n_nodes": n_nodes,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "total_time": elapsed,
        "time_per_step": elapsed / (n_steps * batch_size),
        "steps_per_second": (n_steps * batch_size) / elapsed,
    }

    return stats


def benchmark_coherence(
    n_nodes: int,
    n_trials: int = 100,
    device: str = "cpu",
) -> Dict[str, any]:
    """
    Benchmark coherence and stability.

    Args:
        n_nodes: Number of nodes
        n_trials: Number of trials
        device: Device

    Returns:
        stats: Coherence statistics
    """
    from ..echozero import compute_coherence
    from .builder import build_scaled_system

    # Build system
    K, node_freqs, _, _ = build_scaled_system(n_nodes, device=device)

    system = EchoZeroSystem(
        n_nodes=n_nodes,
        coupling_matrix=K,
        node_freqs=node_freqs,
        device=device,
    )

    coherence_values = []
    stability_count = 0

    for _ in range(n_trials):
        # Random initial state
        psi = torch.complex(
            torch.randn(n_nodes, device=device) * 0.1,
            torch.randn(n_nodes, device=device) * 0.1,
        )

        I_t = torch.zeros(n_nodes, device=device, dtype=torch.complex64)
        desires = torch.randn(n_nodes, device=device)

        # Integrate
        psi_final, _, _ = integrate_echozero(
            psi0=psi,
            t_span=(0.0, 1.0),
            I_t=I_t,
            desires=desires,
            node_freqs=node_freqs,
            K=K,
            device=device,
        )

        # Check stability
        if torch.isfinite(psi_final).all():
            stability_count += 1

        # Compute coherence
        coherence = compute_coherence(psi_final, node_freqs)
        coherence_values.append(coherence.mean().item())

    stats = {
        "n_nodes": n_nodes,
        "n_trials": n_trials,
        "stability_rate": stability_count / n_trials,
        "mean_coherence": np.mean(coherence_values),
        "std_coherence": np.std(coherence_values),
        "min_coherence": np.min(coherence_values),
        "max_coherence": np.max(coherence_values),
    }

    return stats
