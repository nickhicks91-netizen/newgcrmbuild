"""
Harmonic Embedding - Frequency Domain Transformation
Maps grounded input to frequency space with memory-based modulation
"""
import torch
import torch.nn as nn
from typing import Optional


class HarmonicEmbedding(nn.Module):
    """
    Transform input to frequency domain representation.

    Memory context modulates the embedding for narrative continuity.
    Base uses tanh for bounded frequencies, modulation uses sigmoid for gating.
    """

    def __init__(self, input_dim: int, freq_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.freq_dim = freq_dim

        # Base frequency projection
        self.fc = nn.Linear(input_dim, freq_dim)

        # Memory-based modulation (from identity token)
        self.modulator = nn.Linear(32, freq_dim, bias=False)  # memory_size = 32

    def forward(
        self,
        x: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed input into frequency space

        Args:
            x: Input tensor [batch, input_dim]
            memory_context: Identity token from episodic memory [memory_size]

        Returns:
            Frequency embedding [batch, freq_dim]
        """
        # Base frequency embedding (bounded -1 to 1)
        base = torch.tanh(self.fc(x))

        # Apply memory-based modulation if available
        if memory_context is not None:
            batch_size = x.size(0)
            # Expand memory context to batch
            mod_ctx = memory_context.unsqueeze(0).repeat(batch_size, 1)
            # Compute modulation (0 to 1 gating)
            mod = torch.sigmoid(self.modulator(mod_ctx))
            # Modulate base frequencies
            base = base * mod

        return base

    def get_frequency_stats(self, freq: torch.Tensor) -> dict:
        """
        Analyze frequency distribution

        Returns:
            Statistics about frequency embedding
        """
        with torch.no_grad():
            return {
                'mean': freq.mean().item(),
                'std': freq.std().item(),
                'min': freq.min().item(),
                'max': freq.max().item(),
                'variance': freq.var().item()
            }
