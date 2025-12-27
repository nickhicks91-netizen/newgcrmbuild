"""
EchoMirror: Hebbian training for EchoZero coupling matrix.

NO BACKPROPAGATION ALLOWED.

Updates coupling matrix K based on resonance correlation:
    Δk_ij = η * coherence * (ψ_i^* ψ_j + ψ_j^* ψ_i) / 2
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np

from ..echozero.dynamics import enforce_hermiticity


class EchoMirrorTrainer:
    """
    EchoMirror trainer using purely Hebbian learning.

    NO gradient descent. NO backpropagation.
    Updates are based solely on local resonance correlations.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        coherence_threshold: float = 0.3,
        max_coupling: float = 5.0,
        device: str = "cpu",
    ):
        """
        Initialize EchoMirror trainer.

        Args:
            learning_rate: Hebbian learning rate η
            coherence_threshold: Minimum coherence for plasticity
            max_coupling: Maximum coupling strength (stability)
            device: Computation device
        """
        self.learning_rate = learning_rate
        self.coherence_threshold = coherence_threshold
        self.max_coupling = max_coupling
        self.device = device

        # Statistics tracking
        self.update_history = []

    def update_coupling(
        self,
        K: torch.Tensor,
        psi: torch.Tensor,
        coherence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Update coupling matrix using Hebbian rule.

        Δk_ij = η * coherence * (ψ_i^* ψ_j + ψ_j^* ψ_i) / 2

        Args:
            K: Current coupling matrix [n_nodes, n_nodes]
            psi: Resonant state [n_nodes] (complex)
            coherence: Coherence values [n_nodes]

        Returns:
            K_new: Updated coupling matrix
            stats: Update statistics
        """
        # Gate plasticity by coherence threshold
        plasticity_mask = (coherence > self.coherence_threshold).float()

        # Compute Hebbian correlation
        delta_K = hebbian_update(
            psi=psi,
            coherence=coherence,
            learning_rate=self.learning_rate,
        )

        # Apply plasticity mask
        delta_K = delta_K * plasticity_mask.unsqueeze(0) * plasticity_mask.unsqueeze(1)

        # Update coupling
        K_new = K + delta_K

        # Enforce hermiticity
        K_new = enforce_hermiticity(K_new)

        # Clip to prevent instability
        K_new = torch.clamp(K_new.real, -self.max_coupling, self.max_coupling) + \
                1j * torch.clamp(K_new.imag, -self.max_coupling, self.max_coupling)

        # Compute statistics
        stats = {
            "delta_norm": torch.norm(delta_K).item(),
            "coupling_norm": torch.norm(K_new).item(),
            "plasticity_fraction": plasticity_mask.mean().item(),
            "max_coupling": K_new.abs().max().item(),
        }

        self.update_history.append(stats)

        return K_new, stats

    def batch_update(
        self,
        K: torch.Tensor,
        psi_batch: torch.Tensor,
        coherence_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Update coupling using batch of states.

        Args:
            K: Coupling matrix
            psi_batch: Batch of states [batch, n_nodes]
            coherence_batch: Batch of coherence [batch, n_nodes]

        Returns:
            K_new: Updated coupling matrix
            stats: Average statistics
        """
        batch_size = psi_batch.shape[0]

        # Accumulate updates
        delta_K_total = torch.zeros_like(K)
        stats_total = {
            "delta_norm": 0.0,
            "coupling_norm": 0.0,
            "plasticity_fraction": 0.0,
            "max_coupling": 0.0,
        }

        for i in range(batch_size):
            K_new, stats = self.update_coupling(
                K=K,
                psi=psi_batch[i],
                coherence=coherence_batch[i],
            )
            delta_K = K_new - K
            delta_K_total = delta_K_total + delta_K

            for key in stats:
                stats_total[key] += stats[key]

        # Average updates
        delta_K_avg = delta_K_total / batch_size
        K_new = K + delta_K_avg

        # Enforce hermiticity
        K_new = enforce_hermiticity(K_new)

        # Average stats
        for key in stats_total:
            stats_total[key] /= batch_size

        stats_total["coupling_norm"] = torch.norm(K_new).item()

        return K_new, stats_total

    def get_statistics(self) -> Dict[str, any]:
        """Get training statistics."""
        if not self.update_history:
            return {}

        history = {key: [s[key] for s in self.update_history] for key in self.update_history[0]}

        return {
            "n_updates": len(self.update_history),
            "final_delta_norm": history["delta_norm"][-1] if history["delta_norm"] else 0,
            "final_coupling_norm": history["coupling_norm"][-1] if history["coupling_norm"] else 0,
            "avg_plasticity": np.mean(history["plasticity_fraction"]) if history["plasticity_fraction"] else 0,
            "history": history,
        }


def hebbian_update(
    psi: torch.Tensor,
    coherence: torch.Tensor,
    learning_rate: float = 0.001,
) -> torch.Tensor:
    """
    Compute Hebbian coupling update.

    Δk_ij = η * coherence * (ψ_i^* ψ_j + ψ_j^* ψ_i) / 2

    This implements Hebbian learning: "neurons that fire together, wire together"
    For oscillators: "nodes that resonate together, couple together"

    Args:
        psi: Resonant state [n_nodes] (complex)
        coherence: Coherence values [n_nodes]
        learning_rate: Learning rate η

    Returns:
        delta_K: Coupling update [n_nodes, n_nodes]
    """
    n_nodes = psi.shape[0]

    # Compute outer product: ψ_i^* ψ_j
    psi_conj = psi.conj().unsqueeze(1)  # [n_nodes, 1]
    psi_outer = psi.unsqueeze(0)  # [1, n_nodes]

    correlation = psi_conj * psi_outer  # [n_nodes, n_nodes]

    # Symmetrize: (ψ_i^* ψ_j + ψ_j^* ψ_i) / 2
    correlation_sym = (correlation + correlation.conj().T) / 2

    # Weight by coherence (broadcast)
    coherence_weight = coherence.unsqueeze(1) * coherence.unsqueeze(0)  # [n_nodes, n_nodes]

    # Hebbian update
    delta_K = learning_rate * coherence_weight * correlation_sym

    return delta_K


def oja_normalization(
    K: torch.Tensor,
    target_norm: float = 1.0,
) -> torch.Tensor:
    """
    Apply Oja's rule normalization to prevent unbounded growth.

    K_new = K / ||K|| * target_norm

    Args:
        K: Coupling matrix
        target_norm: Target norm

    Returns:
        K_normalized: Normalized coupling matrix
    """
    current_norm = torch.norm(K, p="fro") + 1e-8
    K_normalized = K * (target_norm / current_norm)

    return K_normalized


def stdp_modulation(
    psi_pre: torch.Tensor,
    psi_post: torch.Tensor,
    dt: float = 0.01,
    tau: float = 0.02,
) -> torch.Tensor:
    """
    Spike-timing-dependent plasticity (STDP) modulation.

    For oscillators, we use phase difference as timing signal:
        Δk ∝ exp(-|Δφ| / τ) * sign(Δφ)

    Args:
        psi_pre: Pre-synaptic state (complex)
        psi_post: Post-synaptic state (complex)
        dt: Time difference
        tau: Time constant

    Returns:
        modulation: STDP modulation factor
    """
    # Phase difference
    phase_pre = torch.angle(psi_pre)
    phase_post = torch.angle(psi_post)

    delta_phase = phase_post.unsqueeze(1) - phase_pre.unsqueeze(0)

    # STDP window
    modulation = torch.exp(-torch.abs(delta_phase) / tau) * torch.sign(delta_phase)

    return modulation


def anti_hebbian_update(
    psi: torch.Tensor,
    coherence: torch.Tensor,
    learning_rate: float = 0.001,
) -> torch.Tensor:
    """
    Anti-Hebbian update for decorrelation.

    Δk_ij = -η * coherence * (ψ_i^* ψ_j + ψ_j^* ψ_i) / 2

    Useful for maintaining diversity in resonant patterns.

    Args:
        psi: Resonant state
        coherence: Coherence values
        learning_rate: Learning rate

    Returns:
        delta_K: Anti-Hebbian update
    """
    return -hebbian_update(psi, coherence, learning_rate)
