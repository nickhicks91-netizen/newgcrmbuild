"""
Multi-agent coordinator.

Coordinates multiple governed agents using the global authority allocator.
NO debate, NO voting - only authority-weighted selection.
"""

from typing import Dict, List, Optional, Any
import logging

from .agent import GovernedAgent, AgentConfig
from .allocator import GlobalAuthorityAllocator
from ..core.state import AIState

logger = logging.getLogger(__name__)


class MultiAgentCoordinator:
    """
    Coordinates multiple governed agents.

    Coordination rules:
    - Each agent has local governor
    - Global allocator distributes authority
    - Most stable agent gets priority
    - Unstable agents lose influence smoothly

    NO democratic processes:
    - No voting
    - No majority rule
    - No debate arbitration
    - No consensus seeking
    """

    def __init__(self, blend_rate: float = 0.1, seed: int = 42):
        """
        Initialize coordinator.

        Args:
            blend_rate: Authority reallocation rate
            seed: Random seed
        """
        self.allocator = GlobalAuthorityAllocator(blend_rate=blend_rate, seed=seed)
        self.seed = seed

        # Round counter
        self.round = 0

    def add_agent(self, config: AgentConfig) -> GovernedAgent:
        """
        Add new agent to coordination.

        Args:
            config: Agent configuration

        Returns:
            Created governed agent
        """
        agent = GovernedAgent(config, seed=self.seed)
        self.allocator.register_agent(agent)

        logger.info(f"Added agent: {config.agent_id}")
        return agent

    def remove_agent(self, agent_id: str) -> None:
        """
        Remove agent from coordination.

        Args:
            agent_id: Agent to remove
        """
        self.allocator.unregister_agent(agent_id)
        logger.info(f"Removed agent: {agent_id}")

    def process_round(self, states: Dict[str, AIState]) -> Dict[str, Any]:
        """
        Process one coordination round.

        Args:
            states: Map of agent_id → current state

        Returns:
            Coordination results including authority allocation
        """
        self.round += 1

        # Step 1: Each agent processes its local state
        outputs = {}
        for agent_id, agent in self.allocator.agents.items():
            if agent_id in states:
                output = agent.process_state(states[agent_id])
                outputs[agent_id] = output

        # Step 2: Allocate global authority
        authority_weights = self.allocator.allocate_authority()

        # Step 3: Determine active agent(s)
        active_agents = self.allocator.get_active_agents()
        dominant_agent = self.allocator.get_dominant_agent() if active_agents else None

        # Step 4: Compile results
        results = {
            "round": self.round,
            "authority_weights": authority_weights,
            "active_agents": active_agents,
            "dominant_agent": dominant_agent,
            "outputs": outputs
        }

        logger.info(f"Round {self.round}: {len(active_agents)} active agents, "
                   f"dominant: {dominant_agent}")

        return results

    def select_response(self, agent_responses: Dict[str, str]) -> str:
        """
        Select response based on authority weights.

        Args:
            agent_responses: Map of agent_id → response text

        Returns:
            Selected response (from agent with highest authority)
        """
        if not agent_responses:
            return "[No agents available]"

        # Get dominant agent
        try:
            dominant_agent = self.allocator.get_dominant_agent()
        except ValueError:
            return "[No agents registered]"

        # Return dominant agent's response
        if dominant_agent in agent_responses:
            return agent_responses[dominant_agent]

        # Fallback: return any available response
        return next(iter(agent_responses.values()))

    def get_authority_distribution(self) -> Dict[str, float]:
        """
        Get current authority distribution.

        Returns:
            Map of agent_id → authority weight
        """
        return {
            agent_id: agent.global_authority
            for agent_id, agent in self.allocator.agents.items()
        }

    def detect_hallucination_containment(self) -> bool:
        """
        Check if hallucination is being contained.

        A hallucinating agent should lose authority smoothly.
        This method checks if any unstable agent has low authority.

        Returns:
            True if containment is working
        """
        for agent_id, agent in self.allocator.agents.items():
            stability = agent.get_stability_metric()
            authority = agent.global_authority

            # If agent is very unstable but still has high authority, containment failed
            if stability > 5.0 and authority > 0.5:
                logger.warning(f"Containment failure: {agent_id} unstable but has authority")
                return False

        return True

    def reset(self) -> None:
        """Reset coordinator state."""
        self.allocator.reset()
        self.round = 0
