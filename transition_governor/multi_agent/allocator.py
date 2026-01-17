"""
Global authority allocator for multi-agent systems.

The allocator:
- Collects stability metrics from all agents
- Computes global authority distribution
- Enforces the rule: authority(agent_i) ∝ stability(agent_i)

NO voting, NO majority rule, NO debate arbitration.
Unstable agents lose influence smoothly.
"""

from typing import Dict, List
import logging

from ..core.authority import MultiAgentAuthorityAllocator
from .agent import GovernedAgent

logger = logging.getLogger(__name__)


class GlobalAuthorityAllocator:
    """
    Global authority allocator across agents.

    Core principle:
    - More stable agents get more authority
    - Authority redistribution is smooth
    - No agent can escalate itself
    """

    def __init__(self, blend_rate: float = 0.1, seed: int = 42):
        """
        Initialize allocator.

        Args:
            blend_rate: Rate of authority reallocation
            seed: Random seed
        """
        self.allocator = MultiAgentAuthorityAllocator(
            blend_rate=blend_rate,
            seed=seed
        )
        self.agents: Dict[str, GovernedAgent] = {}

    def register_agent(self, agent: GovernedAgent) -> None:
        """
        Register agent with allocator.

        Args:
            agent: Governed agent to register
        """
        agent_id = agent.config.agent_id
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered, replacing")

        self.agents[agent_id] = agent
        logger.info(f"Registered agent: {agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister agent.

        Args:
            agent_id: Agent to remove
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    def allocate_authority(self) -> Dict[str, float]:
        """
        Allocate authority across all registered agents.

        Returns:
            Map of agent_id → authority weight
        """
        if not self.agents:
            return {}

        # Collect stability metrics from all agents
        stabilities = {
            agent_id: agent.get_stability_metric()
            for agent_id, agent in self.agents.items()
        }

        # Compute global authority allocation
        authority_weights = self.allocator.allocate_global_authority(stabilities)

        # Update each agent's global authority
        for agent_id, authority in authority_weights.items():
            if agent_id in self.agents:
                self.agents[agent_id].set_global_authority(authority)

        # Log allocation
        logger.info(f"Authority allocation: {authority_weights}")

        return authority_weights

    def get_agent_authority(self, agent_id: str) -> float:
        """
        Get current authority for specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Authority weight
        """
        return self.allocator.get_agent_authority(agent_id)

    def get_active_agents(self) -> List[str]:
        """
        Get list of agents with non-zero authority.

        Returns:
            List of agent IDs with authority > threshold
        """
        threshold = 0.1
        return [
            agent_id
            for agent_id, agent in self.agents.items()
            if agent.global_authority >= threshold
        ]

    def get_dominant_agent(self) -> str:
        """
        Get agent with highest authority.

        Returns:
            Agent ID with maximum authority
        """
        if not self.agents:
            raise ValueError("No agents registered")

        return max(
            self.agents.keys(),
            key=lambda aid: self.agents[aid].global_authority
        )

    def reset(self) -> None:
        """Reset allocator and all agents."""
        self.allocator.reset()
        for agent in self.agents.values():
            agent.reset()
