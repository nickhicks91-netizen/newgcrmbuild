"""
Resonant Attention - Coherence-Based Filtering
Implements resonance mechanism: coherence = ReLU(1 - |freq - node_freq| / bandwidth)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..config import AttentionConfig


class ResonantAttention(nn.Module):
    """
    Resonant attention mechanism that filters input based on frequency coherence.

    Formula: coherence = ReLU(1 - |freq_input - node_freq| / bandwidth)

    High coherence (>0.7) indicates resonance and gates memory updates.
    This provides ethical stability by filtering dissonant patterns.
    """

    def __init__(self, freq_dim: int, config: Optional[AttentionConfig] = None):
        super().__init__()
        self.config = config or AttentionConfig(freq_dim=freq_dim)
        self.freq_dim = freq_dim

        # Learnable node frequency (the resonance target)
        self.node_freq = nn.Parameter(torch.randn(freq_dim))

        # Learnable bandwidth parameter
        self.bandwidth = nn.Parameter(torch.tensor(self.config.base_bandwidth))

    def forward(
        self,
        freq_input: torch.Tensor,
        bw_bias: float = 0.0
    ) -> torch.Tensor:
        """
        Compute resonance coherence

        Args:
            freq_input: Frequency embedding [batch, freq_dim]
            bw_bias: Bandwidth bias from desire module (expands during seeking)

        Returns:
            Coherence score [batch, 1] in range [0, 1]
        """
        # Handle tensor or scalar bias
        bias = bw_bias.mean() if hasattr(bw_bias, 'mean') else bw_bias

        # Compute effective bandwidth with bias
        temp_bw = self.bandwidth + bias

        # Clamp bandwidth to valid range
        temp_bw = torch.clamp(temp_bw, min=self.config.bandwidth_min, max=self.config.bandwidth_max)

        # Compute frequency delta
        delta = torch.abs(freq_input - self.node_freq)  # [batch, freq_dim]

        # Resonance formula: coherence = ReLU(1 - delta / bandwidth)
        coherence = F.relu(1 - delta / temp_bw)  # [batch, freq_dim]

        # Average across frequency dimensions
        coherence_score = coherence.mean(dim=-1, keepdim=True)  # [batch, 1]

        return coherence_score

    def is_coherent(self, coherence: torch.Tensor) -> torch.Tensor:
        """
        Check if coherence exceeds threshold

        Args:
            coherence: Coherence scores [batch, 1]

        Returns:
            Boolean mask [batch, 1]
        """
        return (coherence > self.config.coherence_threshold).float()

    def get_resonance_state(self, freq_input: torch.Tensor, bw_bias: float = 0.0) -> dict:
        """
        Get detailed resonance state for analysis

        Returns:
            Dictionary with resonance metrics
        """
        with torch.no_grad():
            coherence = self.forward(freq_input, bw_bias)
            delta = torch.abs(freq_input - self.node_freq)

            return {
                'coherence_mean': coherence.mean().item(),
                'coherence_std': coherence.std().item(),
                'coherent_ratio': self.is_coherent(coherence).mean().item(),
                'avg_freq_delta': delta.mean().item(),
                'bandwidth': (self.bandwidth + bw_bias).item(),
                'node_freq_norm': self.node_freq.norm().item()
            }
