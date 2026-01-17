"""
Desire Module - Goal-Directed Agency
Implements desire vectors that bias attention and gate memory updates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ..config import DesireConfig


class DesireModule(nn.Module):
    """
    Desire-based agency mechanism.

    Maintains learnable desire vectors. Current desire computes alignment
    with frequency input, which:
    1. Gates memory updates (alignment > 0.5)
    2. Biases bandwidth (wider during seeking)

    Alignment formula: cosine_similarity(freq, desire_vec)
    """

    def __init__(self, freq_dim: int, config: Optional[DesireConfig] = None):
        super().__init__()
        self.config = config or DesireConfig()
        self.freq_dim = freq_dim

        # Learnable desire vectors (one per desire type)
        self.desire_vectors = nn.Parameter(
            torch.randn(self.config.num_desires, freq_dim)
        )

        # Current active desire
        self.current_desire_idx = self.config.default_desire_idx

    def forward(self, freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute desire alignment and bandwidth bias

        Args:
            freq: Frequency embedding [batch, freq_dim]

        Returns:
            alignment: Cosine similarity with current desire [batch, 1]
            bandwidth_bias: Bias for attention bandwidth [batch, 1]
        """
        # Get current desire vector
        desire_vec = self.desire_vectors[self.current_desire_idx]  # [freq_dim]

        # Compute alignment (cosine similarity)
        alignment = F.cosine_similarity(
            freq,
            desire_vec.unsqueeze(0),
            dim=-1
        ).unsqueeze(-1)  # [batch, 1]

        # Compute bandwidth bias (positive alignment widens bandwidth for seeking)
        bandwidth_bias = self.config.bandwidth_bias_scale * alignment

        return alignment, bandwidth_bias

    def set_desire(self, desire_idx: int) -> None:
        """
        Switch to a different desire

        Args:
            desire_idx: Index of desire vector (0 to num_desires-1)
        """
        if not 0 <= desire_idx < self.config.num_desires:
            raise ValueError(
                f"desire_idx must be in [0, {self.config.num_desires-1}], got {desire_idx}"
            )
        self.current_desire_idx = desire_idx

    def is_aligned(self, alignment: torch.Tensor) -> torch.Tensor:
        """
        Check if alignment exceeds threshold

        Args:
            alignment: Alignment scores [batch, 1]

        Returns:
            Boolean mask [batch, 1]
        """
        return (alignment > self.config.alignment_threshold).float()

    def get_desire_state(self, freq: torch.Tensor) -> dict:
        """
        Get detailed desire state for analysis

        Returns:
            Dictionary with desire metrics
        """
        with torch.no_grad():
            alignment, bw_bias = self.forward(freq)

            # Compute alignment with all desires
            all_alignments = F.cosine_similarity(
                freq.unsqueeze(1),  # [batch, 1, freq_dim]
                self.desire_vectors.unsqueeze(0),  # [1, num_desires, freq_dim]
                dim=-1
            )  # [batch, num_desires]

            return {
                'current_desire': self.current_desire_idx,
                'alignment_mean': alignment.mean().item(),
                'alignment_std': alignment.std().item(),
                'aligned_ratio': self.is_aligned(alignment).mean().item(),
                'bandwidth_bias_mean': bw_bias.mean().item(),
                'all_alignments_mean': all_alignments.mean(dim=0).tolist(),
                'desire_vec_norm': self.desire_vectors[self.current_desire_idx].norm().item()
            }

    def get_desire_vector(self, desire_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get a desire vector (current or specified)

        Args:
            desire_idx: Desire index, or None for current

        Returns:
            Desire vector [freq_dim]
        """
        idx = desire_idx if desire_idx is not None else self.current_desire_idx
        return self.desire_vectors[idx].detach().clone()
