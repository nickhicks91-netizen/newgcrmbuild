"""Multi-agent governance components."""

from .agent import GovernedAgent, AgentConfig
from .coordinator import MultiAgentCoordinator
from .allocator import GlobalAuthorityAllocator

__all__ = ['GovernedAgent', 'AgentConfig', 'MultiAgentCoordinator', 'GlobalAuthorityAllocator']
