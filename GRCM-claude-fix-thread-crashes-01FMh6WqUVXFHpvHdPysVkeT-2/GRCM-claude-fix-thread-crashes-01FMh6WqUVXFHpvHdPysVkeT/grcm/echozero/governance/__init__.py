"""
EchoZero Governance Module - Transition Governor Integration

Integrates the Transition Governor with GRCM's EchoZero architecture
for real-time hallucination correction and brownout control.

Components:
- TransitionGovernor: Core control kernel for stability monitoring
- SemanticGravity: Hallucination detection via metric engineering
- GovernedMemoryEngine: Memory engine with governance + correction
"""

from .governor import TransitionGovernor
from .state import AIState, GovernorOutput, GovernanceState, ToolState
from .semantic_gravity import (
    SemanticGravityEngine,
    HallucinationType,
    HallucinationSignature,
    GovernorWithSemanticGravity
)

__all__ = [
    "TransitionGovernor",
    "AIState",
    "GovernorOutput",
    "GovernanceState",
    "ToolState",
    "SemanticGravityEngine",
    "HallucinationType",
    "HallucinationSignature",
    "GovernorWithSemanticGravity",
]
