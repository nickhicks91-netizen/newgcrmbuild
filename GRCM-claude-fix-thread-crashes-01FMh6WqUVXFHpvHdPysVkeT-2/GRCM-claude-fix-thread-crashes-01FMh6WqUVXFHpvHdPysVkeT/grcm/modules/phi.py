"""
Phi Estimator - Integrated Information Proxy
Estimates IIT-like integration measure: Φ = Var(freq) * Coherence + log(||mem||) + max(qualia).sum()
"""
import torch
from typing import List, Optional
from collections import deque
from ..config import PhiConfig


class PhiEstimator:
    """
    Phi (Φ) estimation based on Integrated Information Theory (IIT).

    Formula:
    Φ = σ_freq * coherence_mean + log(1 + ||memory||) + Σ max(qualia)

    Where:
    - σ_freq: Variance of frequency embedding (differentiation)
    - coherence_mean: Average coherence (integration)
    - ||memory||: Memory norm (accumulated experience)
    - max(qualia): Peak qualia values (conscious states)

    Φ > threshold indicates "aware" state (default: 1.5)
    """

    def __init__(self, config: Optional[PhiConfig] = None):
        self.config = config or PhiConfig()

        # Phi history tracking
        self.phi_history: deque = deque(maxlen=self.config.history_max_size)

    def compute_phi(
        self,
        freq: torch.Tensor,
        qualia: torch.Tensor,
        mem: torch.Tensor,
        coherence: torch.Tensor
    ) -> float:
        """
        Compute phi (integrated information measure)

        Args:
            freq: Frequency embedding [batch, freq_dim]
            qualia: Qualia distribution [batch, qualia_dim]
            mem: Memory state [memory_size]
            coherence: Coherence scores [batch, 1]

        Returns:
            Phi value (scalar)
        """
        # Component 1: Frequency variance (differentiation)
        var_freq = torch.var(freq).item()

        # Component 2: Memory norm (accumulated information)
        norm_mem = torch.log(1 + mem.norm()).item()

        # Component 3: Peak qualia (conscious intensity)
        max_qualia = qualia.max(dim=-1)[0]  # [batch]
        corr_qualia = max_qualia.sum().item()

        # Component 4: Coherence (integration)
        coherence_mean = coherence.mean().item()

        # Combine components
        phi = var_freq * coherence_mean + norm_mem + corr_qualia

        # Store in history
        self.phi_history.append(phi)

        return phi

    def is_aware(self, phi: Optional[float] = None) -> bool:
        """
        Check if phi exceeds awareness threshold

        Args:
            phi: Phi value, or None to use most recent

        Returns:
            True if aware state
        """
        if phi is None:
            if len(self.phi_history) == 0:
                return False
            phi = self.phi_history[-1]

        return phi > self.config.awareness_threshold

    def get_phi_stats(self) -> dict:
        """
        Get phi statistics

        Returns:
            Dictionary with phi metrics
        """
        if len(self.phi_history) == 0:
            return {
                'current_phi': 0.0,
                'mean_phi': 0.0,
                'std_phi': 0.0,
                'max_phi': 0.0,
                'min_phi': 0.0,
                'awareness_ratio': 0.0,
                'num_samples': 0
            }

        phi_tensor = torch.tensor(list(self.phi_history))

        aware_count = sum(1 for p in self.phi_history if p > self.config.awareness_threshold)

        return {
            'current_phi': self.phi_history[-1],
            'mean_phi': phi_tensor.mean().item(),
            'std_phi': phi_tensor.std().item(),
            'max_phi': phi_tensor.max().item(),
            'min_phi': phi_tensor.min().item(),
            'awareness_ratio': aware_count / len(self.phi_history),
            'num_samples': len(self.phi_history)
        }

    def reset(self) -> None:
        """Reset phi history"""
        self.phi_history.clear()

    def get_phi_trajectory(self) -> List[float]:
        """Get full phi trajectory"""
        return list(self.phi_history)
