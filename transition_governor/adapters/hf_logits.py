"""
HuggingFace LogitsProcessor adapter.

This adapter integrates the Transition Governor with HuggingFace transformers.
It processes logits during generation to enforce governor limits.
"""

import torch
import numpy as np
from typing import Optional, List
from transformers import LogitsProcessor

from ..core.governor import TransitionGovernor
from ..core.state import AIState, ToolState
from ..core.metrics import DirectEntropyAdapter, ConfidenceEstimator


class GovernedLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that enforces Transition Governor constraints.

    This processor:
    1. Computes entropy from logits
    2. Builds AIState
    3. Calls governor
    4. Applies confidence cap to logits
    5. Blocks generation if in brownout

    Integration:
        model.generate(
            ...,
            logits_processor=LogitsProcessorList([
                GovernedLogitsProcessor(governor, ...)
            ])
        )
    """

    def __init__(
        self,
        governor: TransitionGovernor,
        tool_state_provider: Optional['ToolStateProvider'] = None,
        max_context_length: int = 2048,
        seed: int = 42
    ):
        """
        Initialize processor.

        Args:
            governor: TransitionGovernor instance
            tool_state_provider: Provider for tool state (optional)
            max_context_length: Maximum context window size
            seed: Random seed for determinism
        """
        self.governor = governor
        self.tool_state_provider = tool_state_provider
        self.max_context_length = max_context_length
        self.seed = seed

        # Initialize entropy and confidence estimators
        self.entropy_adapter = DirectEntropyAdapter(seed=seed)
        self.confidence_estimator = ConfidenceEstimator()

        # Generation step counter
        self.step = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits with governor constraints.

        Args:
            input_ids: Current generated tokens [batch_size, seq_len]
            scores: Current logits [batch_size, vocab_size]

        Returns:
            Modified logits with governor constraints applied
        """
        # Extract batch (assume single batch for simplicity)
        if scores.shape[0] > 1:
            raise NotImplementedError("Batch processing not yet supported")

        logits = scores[0].cpu().numpy()

        # Step 1: Compute entropy
        entropy = self.entropy_adapter.get_entropy(logits=logits)
        entropy_dot = self.entropy_adapter.get_entropy_dot()

        # Step 2: Compute confidence
        confidence = self.confidence_estimator.estimate_confidence(entropy)
        confidence_dot = self.confidence_estimator.estimate_confidence_dot(
            entropy_dot, confidence
        )

        # Step 3: Get tool state
        tool_state = (
            self.tool_state_provider.current_state()
            if self.tool_state_provider
            else ToolState.INACTIVE
        )

        # Step 4: Get context length
        context_length = input_ids.shape[1]

        # Step 5: Build AIState
        state = AIState(
            entropy=entropy,
            entropy_dot=entropy_dot,
            confidence=confidence,
            confidence_dot=confidence_dot,
            tool_state=tool_state,
            context_length=context_length,
            max_context_length=self.max_context_length,
            fatigue=self.governor.fatigue_accumulator.get_fatigue()
        )

        # Step 6: Call governor
        output = self.governor.govern(state)

        # Step 7: Apply constraints to logits
        modified_logits = self._apply_constraints(logits, output, confidence)

        # Increment step
        self.step += 1

        return torch.tensor(modified_logits, dtype=scores.dtype).unsqueeze(0)

    def _apply_constraints(
        self,
        logits: np.ndarray,
        output: 'GovernorOutput',
        current_confidence: float
    ) -> np.ndarray:
        """
        Apply governor constraints to logits.

        Args:
            logits: Raw logits [vocab_size]
            output: Governor output with constraints
            current_confidence: Current confidence level

        Returns:
            Modified logits
        """
        # Convert to probabilities
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        # If confidence exceeds cap, flatten distribution (increase entropy)
        if current_confidence > output.confidence_cap:
            # Blend with uniform distribution
            vocab_size = len(probs)
            uniform = np.ones(vocab_size) / vocab_size

            # Compute blend factor to achieve target confidence
            # This is a simplified approach; more sophisticated methods possible
            blend_factor = 1.0 - (output.confidence_cap / max(current_confidence, 1e-6))
            blend_factor = np.clip(blend_factor, 0.0, 0.9)  # Don't fully flatten

            probs = (1 - blend_factor) * probs + blend_factor * uniform

        # In brownout, heavily favor safe/generic tokens
        if output.governance_state.value == "brownout":
            # Further flatten distribution
            vocab_size = len(probs)
            uniform = np.ones(vocab_size) / vocab_size
            probs = 0.5 * probs + 0.5 * uniform

        # Convert back to logits
        probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
        modified_logits = np.log(probs)

        return modified_logits

    def reset(self) -> None:
        """Reset processor state."""
        self.step = 0
        self.governor.reset()


class ToolStateTracker:
    """
    Tracks tool execution state for HF generation.

    This is a simple implementation that can be extended to monitor
    actual tool calls in an agent framework.
    """

    def __init__(self):
        self._state = ToolState.INACTIVE
        self._failure_count = 0

    def current_state(self) -> ToolState:
        """Get current tool state."""
        return self._state

    def mark_pending(self) -> None:
        """Mark tool as pending (call initiated)."""
        self._state = ToolState.PENDING

    def mark_active(self) -> None:
        """Mark tool as active (executing)."""
        self._state = ToolState.ACTIVE

    def mark_success(self) -> None:
        """Mark tool call as successful."""
        self._state = ToolState.INACTIVE
        self._failure_count = 0

    def mark_error(self) -> None:
        """Mark tool call as failed."""
        self._state = ToolState.ERROR
        self._failure_count += 1

    def get_failure_count(self) -> int:
        """Get number of consecutive failures."""
        return self._failure_count

    def reset(self) -> None:
        """Reset to inactive."""
        self._state = ToolState.INACTIVE
        self._failure_count = 0
