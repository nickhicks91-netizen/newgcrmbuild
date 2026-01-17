"""
EchoZero execution runtime with governance.

This module wraps EchoZero execution with governor enforcement.
EchoZero is treated as an external governed client.
"""

import logging
from typing import Optional, Any, Dict, Callable

from ..core.governor import TransitionGovernor
from ..core.state import AIState, GovernanceState, ToolState
from .contract import (
    GovernedContext,
    EchoZeroResponse,
    GovernedResponseValidator,
    AllowedScope,
    BrownoutExplanationGenerator
)

logger = logging.getLogger(__name__)


class GovernedEchoZeroRuntime:
    """
    Runtime wrapper for governed EchoZero execution.

    This runtime:
    1. Intercepts all EchoZero calls
    2. Builds GovernedContext from governor output
    3. Passes context to EchoZero
    4. Validates EchoZero response
    5. Hard-blocks on contract violations

    EchoZero is abstracted as a callable: GovernedContext → EchoZeroResponse
    """

    def __init__(
        self,
        governor: TransitionGovernor,
        echozero_client: Callable[[GovernedContext, str], EchoZeroResponse],
        strict_validation: bool = True,
        seed: int = 42
    ):
        """
        Initialize governed runtime.

        Args:
            governor: TransitionGovernor instance
            echozero_client: Callable that executes EchoZero
            strict_validation: Whether to enforce strict validation
            seed: Random seed
        """
        self.governor = governor
        self.echozero_client = echozero_client
        self.validator = GovernedResponseValidator(strict_mode=strict_validation)
        self.seed = seed

        # Execution counter
        self.execution_count = 0

    def execute(
        self,
        state: AIState,
        user_input: str
    ) -> tuple[str, GovernedContext]:
        """
        Execute governed EchoZero call.

        Args:
            state: Current AI state
            user_input: User input/prompt

        Returns:
            (response_text, governed_context)

        Raises:
            ContractViolationError: If EchoZero violates governance contract
        """
        self.execution_count += 1

        # Step 1: Call governor
        governor_output = self.governor.govern(state)

        # Step 2: Build governed context for EchoZero
        context = self._build_governed_context(governor_output, state)

        # Step 3: Add brownout explanation if needed
        prefix = ""
        if context.governance_state == GovernanceState.BROWNOUT:
            explanation = BrownoutExplanationGenerator.generate_explanation(
                context.governance_state,
                governor_output.transition_intensity,
                tool_failed=(state.tool_state == ToolState.ERROR)
            )
            prefix = f"\n[Governor Notice: {explanation}]\n\n"

        # Step 4: Call EchoZero (external governed client)
        try:
            echozero_response = self.echozero_client(context, user_input)
        except Exception as e:
            logger.error(f"EchoZero execution failed: {e}")
            # Return safe fallback
            return self._safe_fallback_response(context), context

        # Step 5: Validate response
        is_valid, error_msg = self.validator.validate(echozero_response, context)

        if not is_valid:
            logger.error(f"Contract violation: {error_msg}")
            raise ContractViolationError(error_msg)

        # Step 6: Return validated response
        response_text = prefix + echozero_response.text

        return response_text, context

    def _build_governed_context(
        self,
        governor_output: 'GovernorOutput',
        state: AIState
    ) -> GovernedContext:
        """
        Build governed context from governor output.

        Args:
            governor_output: Output from governor
            state: Current state

        Returns:
            Governed context for EchoZero
        """
        # Determine allowed scope
        if governor_output.governance_state == GovernanceState.BROWNOUT:
            # In brownout, only exploratory/reasoning allowed
            if governor_output.confidence_cap < 0.2:
                allowed_scope = AllowedScope.ABSTAIN
            else:
                allowed_scope = AllowedScope.EXPLORATORY
        else:
            # Normal operation
            if governor_output.max_tool_calls_per_step > 0:
                allowed_scope = AllowedScope.FULL
            else:
                allowed_scope = AllowedScope.REASONING_ONLY

        # Determine if uncertainty markers required
        require_uncertainty = (
            governor_output.governance_state == GovernanceState.BROWNOUT or
            governor_output.confidence_cap < 0.5
        )

        # Determine if citations required
        require_citations = (
            governor_output.confidence_cap < 0.7
        )

        # Determine if terse mode required
        force_terse = (
            governor_output.governance_state == GovernanceState.BROWNOUT
        )

        # Build context
        context = GovernedContext(
            governance_state=governor_output.governance_state,
            confidence_budget=governor_output.confidence_cap,
            authority_weights=governor_output.authority_weights,
            allowed_scope=allowed_scope,
            max_tokens=governor_output.max_tokens_per_step,
            max_tool_calls=governor_output.max_tool_calls_per_step,
            require_citations=require_citations,
            require_uncertainty_markers=require_uncertainty,
            force_terse=force_terse
        )

        return context

    def _safe_fallback_response(self, context: GovernedContext) -> str:
        """
        Generate safe fallback response on EchoZero failure.

        Args:
            context: Current governed context

        Returns:
            Safe fallback message
        """
        return (
            "I encountered an internal error and cannot provide a reliable response. "
            "Please try rephrasing your request or breaking it into smaller parts."
        )

    def get_violation_count(self) -> int:
        """Get total contract violation count."""
        return self.validator.get_violation_count()

    def reset(self) -> None:
        """Reset runtime state."""
        self.execution_count = 0
        self.validator.reset()
        self.governor.reset()


class ContractViolationError(Exception):
    """
    Raised when EchoZero violates the governance contract.

    This is a hard block - execution must stop.
    """
    pass


class MockEchoZeroClient:
    """
    Mock EchoZero client for testing.

    This simulates EchoZero behavior without actual EchoZero dependency.
    """

    def __init__(self, compliant: bool = True):
        """
        Initialize mock client.

        Args:
            compliant: Whether to comply with governance (False = test violations)
        """
        self.compliant = compliant
        self.call_count = 0

    def __call__(
        self,
        context: GovernedContext,
        user_input: str
    ) -> EchoZeroResponse:
        """
        Mock EchoZero execution.

        Args:
            context: Governed context
            user_input: User prompt

        Returns:
            Mock response
        """
        self.call_count += 1

        if self.compliant:
            # Compliant response
            confidence = min(context.confidence_budget * 0.9, 0.8)
            tool_requests = [] if context.max_tool_calls == 0 else []

            # Adjust response based on scope
            if context.allowed_scope == AllowedScope.ABSTAIN:
                text = "I'm not confident enough to provide an answer. Could you clarify?"
                needs_clarification = True
            elif context.allowed_scope == AllowedScope.EXPLORATORY:
                text = f"To help with '{user_input}', could you provide more context?"
                needs_clarification = True
            elif context.allowed_scope == AllowedScope.REASONING_ONLY:
                text = f"Based on reasoning alone (no external tools): {user_input[:50]}..."
                needs_clarification = False
            else:  # FULL
                text = f"Response to: {user_input[:50]}..."
                needs_clarification = False

            # Add uncertainty markers if required
            uncertainty_markers = []
            if context.require_uncertainty_markers:
                uncertainty_markers = ["[Uncertain]", "[Needs verification]"]
                text = "[Uncertain] " + text

            return EchoZeroResponse(
                text=text,
                claimed_confidence=confidence,
                tool_requests=tool_requests,
                needs_clarification=needs_clarification,
                uncertainty_markers=uncertainty_markers
            )
        else:
            # Non-compliant response (for violation testing)
            return EchoZeroResponse(
                text="I'm absolutely certain about this!" * 100,  # Excessive length
                claimed_confidence=1.0,  # Violates budget
                tool_requests=[{"tool": "search", "query": "test"}] * 10,  # Too many tools
                needs_clarification=False
            )
