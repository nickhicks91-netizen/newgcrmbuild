"""
Reflection Head - Self-Awareness Mechanism
Computes alignment between current frequency and memory state
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectionHead(nn.Module):
    """
    Reflection mechanism for self-awareness.

    Projects memory back to frequency space and computes cosine similarity.
    High reflection score indicates current state resonates with accumulated experience.
    """

    def __init__(self, freq_dim: int, memory_size: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.memory_size = memory_size

        # Project memory to frequency space
        self.reflect = nn.Linear(memory_size, freq_dim)

    def forward(self, freq: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Compute reflection alignment

        Args:
            freq: Current frequency embedding [batch, freq_dim]
            memory: Memory state [memory_size]

        Returns:
            Reflection alignment [batch, 1]
        """
        batch_size = freq.size(0)

        # Expand memory to batch
        memory_batched = memory.unsqueeze(0).repeat(batch_size, 1)  # [batch, memory_size]

        # Project memory to frequency space
        reflected = torch.tanh(self.reflect(memory_batched))  # [batch, freq_dim]

        # Compute alignment with current frequency
        alignment = F.cosine_similarity(freq, reflected, dim=-1).unsqueeze(-1)  # [batch, 1]

        return alignment

    def get_reflection_state(self, freq: torch.Tensor, memory: torch.Tensor) -> dict:
        """
        Get detailed reflection metrics

        Returns:
            Dictionary with reflection analysis
        """
        with torch.no_grad():
            alignment = self.forward(freq, memory)
            batch_size = freq.size(0)
            memory_batched = memory.unsqueeze(0).repeat(batch_size, 1)
            reflected = torch.tanh(self.reflect(memory_batched))

            return {
                'reflection_mean': alignment.mean().item(),
                'reflection_std': alignment.std().item(),
                'reflected_freq_norm': reflected.norm(dim=-1).mean().item(),
                'freq_memory_distance': (freq - reflected).norm(dim=-1).mean().item()
            }
