"""
Memory Grid - Persistent State with Coherence-Gated Updates
Stores resonant patterns using GRU-based integration
"""
import torch
import torch.nn as nn
from typing import Optional
from ..config import MemoryConfig


class MemoryGrid(nn.Module):
    """
    Persistent memory that stores coherent patterns.

    Only updates when coherence > threshold (default 0.7).
    Uses GRU for smooth memory integration over time.
    """

    def __init__(self, memory_size: int, freq_dim: int, config: Optional[MemoryConfig] = None):
        super().__init__()
        self.config = config or MemoryConfig(memory_size=memory_size)
        self.memory_size = memory_size
        self.freq_dim = freq_dim

        # Persistent memory buffer (not trained directly)
        self.memory = nn.Parameter(torch.zeros(memory_size), requires_grad=False)

        # Project frequency signal to memory dimensions
        self.project = nn.Linear(freq_dim, memory_size)

        # GRU for temporal integration
        self.gru = nn.GRUCell(memory_size, memory_size)

        # Statistics tracking
        self.update_count = 0
        self.total_coherence = 0.0

    def update(self, signal: torch.Tensor, coherence: torch.Tensor) -> None:
        """
        Update memory with coherent signals

        Args:
            signal: Frequency embedding [batch, freq_dim]
            coherence: Coherence scores [batch, 1]
        """
        # Create mask for coherent samples (>threshold)
        mask = (coherence > self.config.update_threshold).float()  # [batch, 1]

        # Project signal to memory space
        projected = self.project(signal)  # [batch, memory_size]

        # Weight by coherence mask
        weighted = mask * projected  # [batch, memory_size]

        # Average across batch (imprint)
        imprint = weighted.mean(dim=0)  # [memory_size]

        # Update memory via GRU (smooth integration)
        new_mem = self.gru(
            imprint.unsqueeze(0),
            self.memory.unsqueeze(0)
        ).squeeze(0)

        # Update memory buffer (no gradient tracking)
        self.memory.data.copy_(new_mem)

        # Update statistics
        self.update_count += mask.sum().item()
        self.total_coherence += coherence.sum().item()

    def forward(self) -> torch.Tensor:
        """
        Read current memory state

        Returns:
            Memory vector [memory_size]
        """
        return self.memory

    def reset(self) -> None:
        """Reset memory to zeros"""
        self.memory.data.zero_()
        self.update_count = 0
        self.total_coherence = 0.0

    def get_memory_stats(self) -> dict:
        """
        Get memory statistics

        Returns:
            Dictionary with memory metrics
        """
        with torch.no_grad():
            return {
                'memory_norm': self.memory.norm().item(),
                'memory_mean': self.memory.mean().item(),
                'memory_std': self.memory.std().item(),
                'update_count': self.update_count,
                'avg_coherence': self.total_coherence / max(self.update_count, 1)
            }

    def get_memory_snapshot(self) -> torch.Tensor:
        """Get a detached copy of current memory"""
        return self.memory.detach().clone()
