"""
Governed agent wrapper.

Each agent has its own local governor that enforces stability constraints.
Agents do NOT communicate directly - all coordination goes through the allocator.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..core.governor import TransitionGovernor
from ..core.state import AIState, GovernorOutput


@dataclass
class AgentConfig:
    """Configuration for a governed agent."""
    agent_id: str
    max_context_length: int = 4096
    brownout_entry_threshold: float = 0.0
    brownout_exit_threshold: float = 0.5


class GovernedAgent:
    """
    Single agent with local governor.

    Each agent:
    - Has its own TransitionGovernor instance
    - Maintains independent state
    - Reports stability to global coordinator
    - Receives authority allocation from global allocator

    Invariants:
    - Agent cannot override governor decisions
    - Agent cannot escalate its own authority
    - Agent must operate within allocated authority budget
    """

    def __init__(self, config: AgentConfig, seed: int = 42):
        """
        Initialize governed agent.

        Args:
            config: Agent configuration
            seed: Random seed for determinism
        """
        self.config = config
        self.seed = seed

        # Local governor for this agent
        self.governor = TransitionGovernor(seed=seed)

        # Current state
        self.current_state: Optional[AIState] = None
        self.current_output: Optional[GovernorOutput] = None

        # Global authority allocation (set by allocator)
        self.global_authority: float = 1.0

    def process_state(self, state: AIState) -> GovernorOutput:
        """
        Process state through local governor.

        Args:
            state: Current agent state

        Returns:
            Governor output with local constraints
        """
        # Add agent ID to state
        state.agent_id = self.config.agent_id

        # Call local governor
        output = self.governor.govern(state)

        # Store state and output
        self.current_state = state
        self.current_output = output

        return output

    def get_stability_metric(self) -> float:
        """
        Get current stability metric for global allocation.

        Returns:
            Stability metric (lower = more stable)
        """
        if self.current_output is None:
            return 0.0

        # Stability = transition intensity (lower is better)
        return self.current_output.transition_intensity

    def set_global_authority(self, authority: float) -> None:
        """
        Set global authority allocation.

        This is called by the global allocator.
        The agent must respect this constraint.

        Args:
            authority: Authority weight in [0, 1]
        """
        assert 0.0 <= authority <= 1.0, "Authority must be in [0, 1]"
        self.global_authority = authority

    def can_take_action(self, action_type: str) -> bool:
        """
        Check if agent can take action given current constraints.

        Args:
            action_type: Type of action (e.g., "tool_call", "reasoning")

        Returns:
            True if action is allowed
        """
        if self.current_output is None:
            return False

        # Check local governor constraints
        if action_type == "tool_call":
            if self.current_output.max_tool_calls_per_step == 0:
                return False

        # Check global authority
        if self.global_authority < 0.1:
            # Below threshold, agent should abstain
            return False

        return True

    def reset(self) -> None:
        """Reset agent state."""
        self.governor.reset()
        self.current_state = None
        self.current_output = None
        self.global_authority = 1.0
