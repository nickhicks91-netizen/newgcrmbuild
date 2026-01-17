"""
Resonant Attention - Coherence-Based Filtering (PRODUCTION-HARDENED)
Implements resonance mechanism: coherence = ReLU(1 - |freq - node_freq| / bandwidth)

CHANGES FROM ORIGINAL:
- Added comprehensive error handling
- Added logging throughout
- Added thread safety with RLock
- Added input validation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import threading
from typing import Optional
from ..config import AttentionConfig

logger = logging.getLogger(__name__)


class ResonantAttention(nn.Module):
    """
    Resonant attention mechanism that filters input based on frequency coherence.

    Formula: coherence = ReLU(1 - |freq_input - node_freq| / bandwidth)

    High coherence (>0.7) indicates resonance and gates memory updates.
    This provides ethical stability by filtering dissonant patterns.
    
    Thread-safe: Uses RLock for concurrent access.
    """

    def __init__(self, freq_dim: int, config: Optional[AttentionConfig] = None):
        super().__init__()
        self._lock = threading.RLock()
        
        try:
            logger.info("Initializing ResonantAttention")
            
            self.config = config or AttentionConfig(freq_dim=freq_dim)
            self.freq_dim = freq_dim

            self.node_freq = nn.Parameter(torch.randn(freq_dim))
            self.bandwidth = nn.Parameter(torch.tensor(self.config.base_bandwidth))
            
            logger.info("ResonantAttention initialized successfully")
        except Exception as e:
            logger.error(f"ResonantAttention initialization failed: {e}")
            raise

    def forward(
        self,
        freq_input: torch.Tensor,
        bw_bias: float = 0.0
    ) -> torch.Tensor:
        """
        Compute resonance coherence with error handling.

        Args:
            freq_input: Frequency embedding [batch, freq_dim]
            bw_bias: Bandwidth bias from desire module (expands during seeking)

        Returns:
            Coherence score [batch, 1] in range [0, 1]
        """
        with self._lock:
            try:
                if freq_input is None:
                    raise ValueError("freq_input cannot be None")
                if freq_input.dim() < 2:
                    raise ValueError(f"freq_input must be 2D+, got {freq_input.dim()}D")
                
                bias = bw_bias.mean() if hasattr(bw_bias, 'mean') else bw_bias
                temp_bw = self.bandwidth + bias
                temp_bw = torch.clamp(temp_bw, min=self.config.bandwidth_min, max=self.config.bandwidth_max)
                delta = torch.abs(freq_input - self.node_freq)
                coherence = F.relu(1 - delta / temp_bw)
                coherence_score = coherence.mean(dim=-1, keepdim=True)

                return coherence_score
                
            except ValueError as e:
                logger.error(f"Attention validation failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Attention forward failed: {e}")
                raise

    def is_coherent(self, coherence: torch.Tensor) -> torch.Tensor:
        """
        Check if coherence exceeds threshold

        Args:
            coherence: Coherence scores [batch, 1]

        Returns:
            Boolean mask [batch, 1]
        """
        with self._lock:
            try:
                return (coherence > self.config.coherence_threshold).float()
            except Exception as e:
                logger.error(f"is_coherent failed: {e}")
                raise

    def get_resonance_state(self, freq_input: torch.Tensor, bw_bias: float = 0.0) -> dict:
        """
        Get detailed resonance state for analysis

        Returns:
            Dictionary with resonance metrics
        """
        with self._lock:
            try:
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
            except Exception as e:
                logger.error(f"get_resonance_state failed: {e}")
                raise
