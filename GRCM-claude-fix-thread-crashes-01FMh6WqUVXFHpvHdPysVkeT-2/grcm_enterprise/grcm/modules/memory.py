"""
Memory Grid - Persistent State with Coherence-Gated Updates (PRODUCTION-HARDENED)
Stores resonant patterns using GRU-based integration

CHANGES FROM ORIGINAL:
- Added comprehensive error handling
- Added logging throughout
- Added thread safety with RLock
- Added input validation
"""
import torch
import torch.nn as nn
import logging
import threading
from typing import Optional
from ..config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryGrid(nn.Module):
    """
    Persistent memory that stores coherent patterns.

    Only updates when coherence > threshold (default 0.7).
    Uses GRU for smooth memory integration over time.
    
    Thread-safe: Uses RLock for concurrent access.
    """

    def __init__(self, memory_size: int, freq_dim: int, config: Optional[MemoryConfig] = None):
        super().__init__()
        self._lock = threading.RLock()
        
        try:
            logger.info("Initializing MemoryGrid")
            
            self.config = config or MemoryConfig(memory_size=memory_size)
            self.memory_size = memory_size
            self.freq_dim = freq_dim

            self.memory = nn.Parameter(torch.zeros(memory_size), requires_grad=False)
            self.project = nn.Linear(freq_dim, memory_size)
            self.gru = nn.GRUCell(memory_size, memory_size)

            self.update_count = 0
            self.total_coherence = 0.0
            
            logger.info("MemoryGrid initialized successfully")
        except Exception as e:
            logger.error(f"MemoryGrid initialization failed: {e}")
            raise

    def update(self, signal: torch.Tensor, coherence: torch.Tensor) -> None:
        """
        Thread-safe update memory with coherent signals

        Args:
            signal: Frequency embedding [batch, freq_dim]
            coherence: Coherence scores [batch, 1]
        """
        with self._lock:
            try:
                if signal is None:
                    raise ValueError("signal cannot be None")
                if coherence is None:
                    raise ValueError("coherence cannot be None")
                
                mask = (coherence > self.config.update_threshold).float()
                projected = self.project(signal)
                weighted = mask * projected
                imprint = weighted.mean(dim=0)

                new_mem = self.gru(
                    imprint.unsqueeze(0),
                    self.memory.unsqueeze(0)
                ).squeeze(0)

                self.memory.data.copy_(new_mem)
                self.update_count += mask.sum().item()
                self.total_coherence += coherence.sum().item()
                
                logger.debug("Memory updated successfully")
            except ValueError as e:
                logger.error(f"Memory update validation failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Memory update failed: {e}")
                raise

    def forward(self) -> torch.Tensor:
        """
        Read current memory state

        Returns:
            Memory vector [memory_size]
        """
        with self._lock:
            return self.memory

    def reset(self) -> None:
        """Thread-safe reset memory to zeros"""
        with self._lock:
            try:
                self.memory.data.zero_()
                self.update_count = 0
                self.total_coherence = 0.0
                logger.info("Memory reset successfully")
            except Exception as e:
                logger.error(f"Memory reset failed: {e}")
                raise

    def get_memory_stats(self) -> dict:
        """
        Get memory statistics

        Returns:
            Dictionary with memory metrics
        """
        with self._lock:
            try:
                with torch.no_grad():
                    return {
                        'memory_norm': self.memory.norm().item(),
                        'memory_mean': self.memory.mean().item(),
                        'memory_std': self.memory.std().item(),
                        'update_count': self.update_count,
                        'avg_coherence': self.total_coherence / max(self.update_count, 1)
                    }
            except Exception as e:
                logger.error(f"get_memory_stats failed: {e}")
                raise

    def get_memory_snapshot(self) -> torch.Tensor:
        """Get a detached copy of current memory"""
        with self._lock:
            return self.memory.detach().clone()
