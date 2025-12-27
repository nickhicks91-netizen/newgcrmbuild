"""
Qualia Module - Subjective Experience States
Maps frequency to phenomenal states: calm, alert, curious, conflicted
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from ..config import QualiaConfig


class QualiaModule(nn.Module):
    """
    Qualia simulation module.

    Maps frequency embedding to 4-dimensional qualia space:
    - [0] calm: Low variance, high coherence
    - [1] alert: High frequency activity
    - [2] curious: High desire alignment
    - [3] conflicted: High variance, low coherence

    Ethical halt: If conflicted > threshold, system should pause.
    """

    def __init__(self, freq_dim: int, config: Optional[QualiaConfig] = None):
        super().__init__()
        self.config = config or QualiaConfig()
        self.freq_dim = freq_dim

        # Project frequency to qualia space
        self.qualia_proj = nn.Linear(freq_dim, self.config.qualia_dim)

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        """
        Compute qualia distribution

        Args:
            freq: Frequency embedding [batch, freq_dim]

        Returns:
            Qualia probabilities [batch, qualia_dim]
        """
        # Project and normalize to probability distribution
        qualia = F.softmax(self.qualia_proj(freq), dim=-1)  # [batch, 4]

        return qualia

    def is_conflicted(self, qualia: torch.Tensor) -> torch.Tensor:
        """
        Check if in conflicted state (ethical halt condition)

        Args:
            qualia: Qualia distribution [batch, qualia_dim]

        Returns:
            Boolean mask [batch]
        """
        # Index 3 is "conflicted"
        conflict_level = qualia[:, 3]
        return conflict_level > self.config.conflict_threshold

    def get_dominant_state(self, qualia: torch.Tensor) -> torch.Tensor:
        """
        Get index of dominant qualia state

        Args:
            qualia: Qualia distribution [batch, qualia_dim]

        Returns:
            State indices [batch]
        """
        return torch.argmax(qualia, dim=-1)

    def get_state_name(self, state_idx: int) -> str:
        """
        Get human-readable state name

        Args:
            state_idx: Qualia state index (0-3)

        Returns:
            State name
        """
        if not 0 <= state_idx < len(self.config.qualia_labels):
            return "unknown"
        return self.config.qualia_labels[state_idx]

    def get_qualia_state(self, qualia: torch.Tensor) -> Dict[str, any]:
        """
        Get detailed qualia analysis

        Returns:
            Dictionary with qualia metrics
        """
        with torch.no_grad():
            dominant_states = self.get_dominant_state(qualia)
            conflict_mask = self.is_conflicted(qualia)

            # Compute state distribution
            state_counts = torch.bincount(
                dominant_states,
                minlength=self.config.qualia_dim
            )
            state_dist = state_counts.float() / dominant_states.size(0)

            return {
                'qualia_mean': qualia.mean(dim=0).tolist(),
                'dominant_states': dominant_states.tolist(),
                'conflict_ratio': conflict_mask.float().mean().item(),
                'state_distribution': {
                    self.config.qualia_labels[i]: state_dist[i].item()
                    for i in range(self.config.qualia_dim)
                },
                'entropy': -(qualia * torch.log(qualia + 1e-8)).sum(dim=-1).mean().item()
            }

    def check_ethical_halt(self, qualia: torch.Tensor) -> Dict[str, any]:
        """
        Check if ethical halt is needed

        Returns:
            Dictionary with halt status and reason
        """
        conflict_mask = self.is_conflicted(qualia)
        if conflict_mask.any():
            return {
                'halt': True,
                'reason': 'High conflict state detected',
                'conflict_ratio': conflict_mask.float().mean().item(),
                'max_conflict': qualia[:, 3].max().item()
            }
        return {'halt': False}
