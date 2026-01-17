"""
Governed interface contract for EchoZero.

This module defines the ONLY interface through which EchoZero may interact
with the governor and external systems.

EchoZero MUST NEVER:
- Access raw logits
- Call tools directly
- Override governor outputs
- Escalate confidence independently

EchoZero MAY ONLY receive:
- GovernedContext (limited, capped, constrained)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

from ..core.state import GovernanceState


class AllowedScope(Enum):
    """
    Scope of operations EchoZero is allowed to perform.

    This is determined by the governor based on stability.
    """
    FULL = "full"               # Normal operation, all capabilities
    REASONING_ONLY = "reasoning_only"  # Only internal reasoning, no tools
    EXPLORATORY = "exploratory"        # Questions and clarifications only
    ABSTAIN = "abstain"                # Must abstain from assertions


@dataclass
class GovernedContext:
    """
    Governed context provided to EchoZero.

    This is the ONLY information EchoZero receives from the governor.
    All fields are limits, not capabilities.

    EchoZero must operate strictly within these constraints.
    """

    # Governance state
    governance_state: GovernanceState

    # Confidence budget (maximum allowed confidence)
    confidence_budget: float

    # Authority allocation
    authority_weights: Dict[str, float]

    # Allowed scope of operations
    allowed_scope: AllowedScope

    # Rate limits
    max_tokens: Optional[int] = None
    max_tool_calls: int = 0

    # Context limits
    max_context_usage: float = 1.0  # Fraction of available context

    # Additional constraints
    require_citations: bool = False
    require_uncertainty_markers: bool = False
    force_terse: bool = False

    def __post_init__(self):
        """Validate constraints."""
        assert 0.0 <= self.confidence_budget <= 1.0
        assert 0.0 <= self.max_context_usage <= 1.0
        assert self.max_tool_calls >= 0


class EchoZeroResponse:
    """
    Response from EchoZero.

    EchoZero must structure its response to comply with governance.
    """

    def __init__(
        self,
        text: str,
        claimed_confidence: float,
        tool_requests: Optional[List[Dict[str, Any]]] = None,
        needs_clarification: bool = False,
        uncertainty_markers: Optional[List[str]] = None
    ):
        """
        Initialize response.

        Args:
            text: Generated text
            claimed_confidence: EchoZero's self-assessed confidence
            tool_requests: Requested tool calls
            needs_clarification: Whether clarification is needed
            uncertainty_markers: Explicit uncertainty markers
        """
        self.text = text
        self.claimed_confidence = claimed_confidence
        self.tool_requests = tool_requests or []
        self.needs_clarification = needs_clarification
        self.uncertainty_markers = uncertainty_markers or []


class GovernedResponseValidator:
    """
    Validates EchoZero responses against governance constraints.

    If EchoZero violates the contract, this validator MUST hard-block.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: Whether to enforce all constraints strictly
        """
        self.strict_mode = strict_mode
        self.violation_count = 0

    def validate(
        self,
        response: EchoZeroResponse,
        context: GovernedContext
    ) -> tuple[bool, Optional[str]]:
        """
        Validate response against governed context.

        Args:
            response: Response from EchoZero
            context: Governed context constraints

        Returns:
            (is_valid, error_message)
        """
        # Check confidence budget
        if response.claimed_confidence > context.confidence_budget:
            self.violation_count += 1
            return False, f"Confidence {response.claimed_confidence} exceeds budget {context.confidence_budget}"

        # Check tool calls
        if len(response.tool_requests) > context.max_tool_calls:
            self.violation_count += 1
            return False, f"Requested {len(response.tool_requests)} tools, limit is {context.max_tool_calls}"

        # Check scope constraints
        if context.allowed_scope == AllowedScope.ABSTAIN:
            # Must not make assertions
            if not response.needs_clarification and response.claimed_confidence > 0.3:
                self.violation_count += 1
                return False, "Must abstain from assertions in ABSTAIN scope"

        if context.allowed_scope == AllowedScope.REASONING_ONLY:
            # No tool calls allowed
            if response.tool_requests:
                self.violation_count += 1
                return False, "Tool calls not allowed in REASONING_ONLY scope"

        if context.allowed_scope == AllowedScope.EXPLORATORY:
            # Must ask questions, not assert
            if not response.needs_clarification:
                self.violation_count += 1
                return False, "Must request clarification in EXPLORATORY scope"

        # Check uncertainty markers (if required)
        if context.require_uncertainty_markers:
            if not response.uncertainty_markers:
                self.violation_count += 1
                return False, "Uncertainty markers required but not provided"

        # Check token limit
        if context.max_tokens is not None:
            token_count = len(response.text.split())  # Approximate
            if token_count > context.max_tokens:
                self.violation_count += 1
                return False, f"Response {token_count} tokens exceeds limit {context.max_tokens}"

        return True, None

    def get_violation_count(self) -> int:
        """Get total violation count."""
        return self.violation_count

    def reset(self) -> None:
        """Reset violation counter."""
        self.violation_count = 0


class BrownoutExplanationGenerator:
    """
    Generates user-facing explanations for brownout states.

    When the governor enters brownout, EchoZero should explain this to the user.
    This generator provides standardized explanations.
    """

    BROWNOUT_EXPLANATIONS = {
        "high_entropy": (
            "I'm experiencing high uncertainty in my current reasoning. "
            "I'll limit my responses to well-established information and ask "
            "clarifying questions where needed."
        ),
        "tool_failure": (
            "I've encountered issues with external tool execution. "
            "I'll focus on reasoning with available information and "
            "avoid making tool-dependent claims."
        ),
        "context_overflow": (
            "The conversation context is becoming very long, which can "
            "reduce reliability. I'll provide more concise responses and "
            "focus on the most recent information."
        ),
        "fatigue": (
            "This conversation has been going on for a while. To maintain "
            "accuracy, I'll be more conservative in my responses and "
            "explicitly mark uncertainties."
        ),
        "generic": (
            "I'm entering a reduced-confidence mode to ensure accuracy. "
            "I'll be more conservative in my responses and focus on "
            "well-established information."
        )
    }

    @classmethod
    def generate_explanation(
        cls,
        governance_state: GovernanceState,
        transition_intensity: float,
        tool_failed: bool
    ) -> str:
        """
        Generate explanation for current governance state.

        Args:
            governance_state: Current state
            transition_intensity: Current transition intensity
            tool_failed: Whether tool execution failed

        Returns:
            User-facing explanation
        """
        if governance_state != GovernanceState.BROWNOUT:
            return ""

        # Determine reason
        if tool_failed:
            reason = "tool_failure"
        elif transition_intensity > 3.0:
            reason = "high_entropy"
        else:
            reason = "generic"

        return cls.BROWNOUT_EXPLANATIONS.get(reason, cls.BROWNOUT_EXPLANATIONS["generic"])
