"""
Episodic Threading - Narrative Identity Construction
Maintains episodic memory and evolves identity token via GRU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, List, Tuple
from ..config import ThreadingConfig


class EpisodicThreadBank:
    """
    Episodic memory bank that maintains narrative identity.

    Stores episodes as (timestamp, memory, qualia) tuples.
    Identity token evolves via GRU based on combined memory + qualia.
    Arc bias computed from recent vs historical qualia similarity.
    """

    def __init__(
        self,
        memory_size: int = 32,
        qualia_dim: int = 4,
        config: Optional[ThreadingConfig] = None
    ):
        self.config = config or ThreadingConfig()
        self.memory_size = memory_size
        self.qualia_dim = qualia_dim

        # Episodic storage (circular buffer)
        self.threads = deque(maxlen=self.config.max_episodes)

        # Identity token (evolves over time)
        self.identity_token = torch.zeros(memory_size)

        # GRU for identity evolution
        self.gru_id = nn.GRUCell(memory_size, memory_size)

    def add_episode(
        self,
        timestamp: int,
        mem: torch.Tensor,
        qualia: torch.Tensor
    ) -> None:
        """
        Add new episode to memory bank

        Args:
            timestamp: Episode timestamp
            mem: Memory state [memory_size]
            qualia: Qualia distribution [qualia_dim] or [batch, qualia_dim]
        """
        # Average qualia across batch if needed
        if qualia.dim() > 1:
            qualia_avg = qualia.mean(0)
        else:
            qualia_avg = qualia

        # Pad qualia to memory_size for combination
        qualia_proj = F.pad(qualia_avg, (0, self.memory_size - self.qualia_dim))

        # Combine memory and qualia
        combined = mem + qualia_proj

        # Store episode
        self.threads.append((timestamp, mem.clone(), qualia_avg.clone()))

        # Evolve identity token
        new_id = self.gru_id(
            combined.unsqueeze(0),
            self.identity_token.unsqueeze(0)
        ).squeeze(0)

        self.identity_token = new_id.detach()  # No gradient tracking for identity

    def get_arc(self, current_coherence: float, freq_dim: int) -> torch.Tensor:
        """
        Compute narrative arc bias

        Arc measures similarity between recent and historical qualia,
        weighted by current coherence. Represents narrative pull/continuity.

        Formula: arc_delta = cos_sim(recent_qualia, historical_mean) * coherence

        Args:
            current_coherence: Current coherence score
            freq_dim: Frequency dimension

        Returns:
            Arc bias vector [freq_dim]
        """
        if len(self.threads) < self.config.min_episodes_for_arc:
            return torch.zeros(freq_dim)

        # Get recent qualia
        recent = self.threads[-1][2]  # Last episode's qualia

        # Get historical qualia (all except most recent)
        historical_qualia = [th[2] for th in list(self.threads)[:-1]]

        if len(historical_qualia) == 0:
            historical = torch.zeros_like(recent)
        else:
            historical = torch.stack(historical_qualia).mean(0)

        # Compute arc as cosine similarity
        arc_delta = F.cosine_similarity(
            recent.unsqueeze(0),
            historical.unsqueeze(0),
            dim=-1
        ).item() * current_coherence

        # Convert to bias vector
        arc_bias = arc_delta * torch.ones(freq_dim) * self.config.arc_scale

        return arc_bias

    def get_identity_token(self) -> torch.Tensor:
        """Get current identity token"""
        return self.identity_token.clone()

    def get_episodes(self) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
        """Get all stored episodes"""
        return list(self.threads)

    def reset(self) -> None:
        """Reset all episodic memory"""
        self.threads.clear()
        self.identity_token.zero_()

    def get_threading_stats(self) -> dict:
        """
        Get episodic memory statistics

        Returns:
            Dictionary with threading metrics
        """
        if len(self.threads) == 0:
            return {
                'num_episodes': 0,
                'identity_norm': self.identity_token.norm().item(),
                'identity_mean': self.identity_token.mean().item()
            }

        timestamps = [ep[0] for ep in self.threads]
        memories = torch.stack([ep[1] for ep in self.threads])
        qualias = torch.stack([ep[2] for ep in self.threads])

        return {
            'num_episodes': len(self.threads),
            'time_span': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'identity_norm': self.identity_token.norm().item(),
            'identity_mean': self.identity_token.mean().item(),
            'identity_std': self.identity_token.std().item(),
            'memory_trajectory_var': memories.var(dim=0).mean().item(),
            'qualia_trajectory_var': qualias.var(dim=0).mean().item()
        }
