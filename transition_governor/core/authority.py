"""
Authority redistribution logic.

This module implements authority reallocation based on stability.
Authority is a budget allocation, not a permission system.

Core principle:
- Unstable subsystems lose authority smoothly
- Stable subsystems gain authority proportionally
- Total authority always sums to 1.0
"""

import numpy as np
from typing import Dict
from .state import ToolState


class AuthorityAllocator:
    """
    Deterministic authority redistribution.

    Authority weights control resource allocation across subsystems:
    - reasoning: base model output
    - tools: external tool calls
    - memory: context/retrieval usage

    Invariants:
    - Sum of weights always equals 1.0
    - Weights are always non-negative
    - Redistribution is smooth and rate-limited
    """

    # Default authority distribution (normal operation)
    DEFAULT_WEIGHTS = {
        "reasoning": 0.5,
        "tools": 0.3,
        "memory": 0.2,
    }

    # Brownout authority distribution (reduced capability)
    BROWNOUT_WEIGHTS = {
        "reasoning": 0.9,  # Favor safe, local reasoning
        "tools": 0.0,      # Zero tool authority (no external actions)
        "memory": 0.1,     # Minimal memory (reduce context risk)
    }

    def __init__(self, blend_rate: float = 0.1, seed: int = 42):
        """
        Initialize allocator.

        Args:
            blend_rate: Rate of authority reallocation (0 = no change, 1 = instant)
            seed: Random seed for deterministic behavior
        """
        self.blend_rate = blend_rate
        self.seed = seed
        self.current_weights = self.DEFAULT_WEIGHTS.copy()
        np.random.seed(seed)

    def reallocate(
        self,
        transition_intensity: float,
        tool_state: ToolState,
        in_brownout: bool
    ) -> Dict[str, float]:
        """
        Reallocate authority based on system state.

        Args:
            transition_intensity: Magnitude of current transition
            tool_state: Current tool execution state
            in_brownout: Whether system is in brownout mode

        Returns:
            New authority weights (sum to 1.0)
        """
        # Determine target weights
        if in_brownout:
            target_weights = self.BROWNOUT_WEIGHTS.copy()
        else:
            target_weights = self._compute_normal_weights(
                transition_intensity, tool_state
            )

        # Blend current weights toward target (smooth transition)
        new_weights = {}
        for key in self.DEFAULT_WEIGHTS.keys():
            current = self.current_weights[key]
            target = target_weights[key]
            new_weights[key] = current + self.blend_rate * (target - current)

        # Normalize to ensure sum is exactly 1.0
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        self.current_weights = new_weights
        return new_weights

    def _compute_normal_weights(
        self,
        transition_intensity: float,
        tool_state: ToolState
    ) -> Dict[str, float]:
        """
        Compute target authority weights for normal operation.

        Args:
            transition_intensity: Current transition magnitude
            tool_state: Tool execution state

        Returns:
            Target weights
        """
        weights = self.DEFAULT_WEIGHTS.copy()

        # Reduce tool authority if tool is in error state
        if tool_state == ToolState.ERROR:
            weights["tools"] *= 0.5
            # Redistribute to reasoning
            weights["reasoning"] += weights["tools"] * 0.5

        # Reduce tool authority under high transition intensity
        if transition_intensity > 1.0:
            reduction_factor = min(1.0, transition_intensity / 2.0)
            weights["tools"] *= (1.0 - reduction_factor)
            # Redistribute to reasoning and memory
            weights["reasoning"] += weights["tools"] * reduction_factor * 0.7
            weights["memory"] += weights["tools"] * reduction_factor * 0.3

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def reset(self) -> None:
        """Reset to default weights."""
        self.current_weights = self.DEFAULT_WEIGHTS.copy()


class MultiAgentAuthorityAllocator:
    """
    Global authority allocation across multiple agents.

    Core rule:
        authority(agent_i) ∝ stability(agent_i)

    Unstable agents lose influence smoothly.
    No voting, no majority rule, no debate.
    """

    def __init__(self, blend_rate: float = 0.1, seed: int = 42):
        """
        Initialize multi-agent allocator.

        Args:
            blend_rate: Rate of authority reallocation
            seed: Random seed for determinism
        """
        self.blend_rate = blend_rate
        self.seed = seed
        self.agent_weights: Dict[str, float] = {}
        np.random.seed(seed)

    def allocate_global_authority(
        self,
        agent_stabilities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Allocate authority across agents based on stability.

        Args:
            agent_stabilities: Map of agent_id → stability metric (lower is better)

        Returns:
            Map of agent_id → authority weight (sum to 1.0)
        """
        if not agent_stabilities:
            return {}

        # Convert stability to authority (inverse relationship)
        # More stable (lower stability metric) = higher authority
        max_stability = max(agent_stabilities.values())
        authority_scores = {
            agent_id: max_stability - stability + 1e-6  # Add epsilon to avoid zero
            for agent_id, stability in agent_stabilities.items()
        }

        # Normalize to sum to 1.0
        total = sum(authority_scores.values())
        target_weights = {
            agent_id: score / total
            for agent_id, score in authority_scores.items()
        }

        # Blend with current weights (smooth transition)
        if not self.agent_weights:
            # First allocation: use target directly
            self.agent_weights = target_weights
        else:
            new_weights = {}
            for agent_id in target_weights:
                current = self.agent_weights.get(agent_id, 0.0)
                target = target_weights[agent_id]
                new_weights[agent_id] = current + self.blend_rate * (target - current)

            # Normalize
            total = sum(new_weights.values())
            self.agent_weights = {k: v / total for k, v in new_weights.items()}

        return self.agent_weights.copy()

    def get_agent_authority(self, agent_id: str) -> float:
        """
        Get current authority for specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Authority weight (0.0 to 1.0)
        """
        return self.agent_weights.get(agent_id, 0.0)

    def reset(self) -> None:
        """Reset all agent weights."""
        self.agent_weights.clear()
