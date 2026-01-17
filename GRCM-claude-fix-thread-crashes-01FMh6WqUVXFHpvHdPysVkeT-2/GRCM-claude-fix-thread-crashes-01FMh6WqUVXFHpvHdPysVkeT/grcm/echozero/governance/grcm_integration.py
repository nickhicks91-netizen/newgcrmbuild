"""
GRCM Integration Layer for Transition Governor + Semantic Gravity

This module provides integration between:
- GRCM's ModularGRCM architecture
- Transition Governor for stability monitoring
- Semantic Gravity for hallucination correction
- EchoZero memory systems

Usage:
    governed_grcm = GovernedGRCM(config)
    output = governed_grcm(input_data)
    # Automatically monitors stability and corrects hallucinations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .governor import TransitionGovernor
from .state import AIState, GovernorOutput, GovernanceState, ToolState
from .semantic_gravity import SemanticGravityEngine, GovernorWithSemanticGravity


@dataclass
class GRCMState:
    """
    Bridge between GRCM outputs and Governor inputs.

    Converts GRCM's resonance/qualia metrics into AIState format.
    """
    coherence: float  # From ResonantAttention
    alignment: float  # From DesireModule
    reflection_norm: float  # From ReflectionHead
    qualia_conflicted: float  # From QualiaModule
    memory_utilization: float  # From MemoryGrid

    def to_ai_state(self, prev_state: Optional['GRCMState'] = None) -> AIState:
        """
        Convert GRCM state to AIState for governor.

        Mapping:
        - entropy ← (1 - coherence) * 4.0  # Low coherence = high entropy
        - confidence ← alignment  # High alignment = high confidence
        - fatigue ← memory_utilization
        - tool_state ← INACTIVE (GRCM doesn't use tools in this version)
        """
        # Compute entropy from coherence
        entropy = (1.0 - self.coherence) * 4.0  # Scale to [0, 4] bits

        # Compute rates if we have previous state
        if prev_state:
            entropy_dot = entropy - ((1.0 - prev_state.coherence) * 4.0)
            confidence_dot = self.alignment - prev_state.alignment
        else:
            entropy_dot = 0.0
            confidence_dot = 0.0

        return AIState(
            entropy=entropy,
            entropy_dot=entropy_dot,
            confidence=self.alignment,
            confidence_dot=confidence_dot,
            tool_state=ToolState.INACTIVE,
            context_length=0,  # Not tracked in GRCM
            max_context_length=2048,
            fatigue=self.memory_utilization
        )


class GovernedMemoryIntegration:
    """
    Integrates Governor + Semantic Gravity with EchoZero memory engine.

    Wraps GRCM's memory systems to provide:
    - Real-time stability monitoring
    - Brownout detection
    - Hallucination correction via semantic gravity
    """

    def __init__(
        self,
        memory_engine,  # EchoZeroMemoryEngine instance
        enable_governance: bool = True,
        enable_correction: bool = True,
        brownout_threshold: float = 2.0
    ):
        """
        Args:
            memory_engine: Existing EchoZero memory engine
            enable_governance: Enable transition governor monitoring
            enable_correction: Enable semantic gravity correction
            brownout_threshold: Transition intensity threshold for brownout
        """
        self.memory_engine = memory_engine
        self.enable_governance = enable_governance
        self.enable_correction = enable_correction

        if enable_governance:
            self.governor = TransitionGovernor(seed=42)

        if enable_correction:
            # Create semantic gravity engine
            # Note: Requires holographic memory interface
            # This will be implemented when GRCM adopts holographic architecture
            self.gravity_engine = None  # Placeholder for future integration

        self.prev_grcm_state: Optional[GRCMState] = None

    def process_with_governance(
        self,
        state_vector: np.ndarray,
        grcm_metrics: Dict[str, float],
        torsion: float = 1.0
    ) -> Tuple[Dict[str, Any], GovernorOutput]:
        """
        Process state through memory with governance.

        Args:
            state_vector: State to store in memory
            grcm_metrics: Dict with keys: coherence, alignment, reflection_norm,
                         qualia_conflicted, memory_utilization
            torsion: Importance score

        Returns:
            (memory_routing_summary, governor_output)
        """
        # Convert GRCM metrics to governor format
        grcm_state = GRCMState(**grcm_metrics)
        ai_state = grcm_state.to_ai_state(self.prev_grcm_state)

        # Run governance check
        if self.enable_governance:
            gov_output = self.governor.govern(ai_state)
        else:
            gov_output = None

        # Process through memory
        memory_result = self.memory_engine.process(
            state=state_vector,
            torsion=torsion,
            metadata={'governance': gov_output}
        )

        # Store state for next iteration
        self.prev_grcm_state = grcm_state

        return memory_result, gov_output

    def should_brownout(self, gov_output: GovernorOutput) -> bool:
        """Check if system should enter brownout state."""
        return gov_output.governance_state == GovernanceState.BROWNOUT if gov_output else False


def add_governance_to_grcm(grcm_module, enable_governance=True, enable_correction=True):
    """
    Factory function to add governance to existing GRCM instance.

    Args:
        grcm_module: ModularGRCM instance
        enable_governance: Enable transition governor
        enable_correction: Enable semantic gravity correction

    Returns:
        Modified GRCM with governance capabilities
    """
    # Wrap the memory module
    if hasattr(grcm_module, 'memory'):
        # Check if memory has EchoZero engine
        if hasattr(grcm_module.memory, 'echozero_engine'):
            grcm_module.governed_memory = GovernedMemoryIntegration(
                memory_engine=grcm_module.memory.echozero_engine,
                enable_governance=enable_governance,
                enable_correction=enable_correction
            )

    return grcm_module


class GovernedGRCMForward:
    """
    Governance-aware forward pass for GRCM.

    Replaces standard forward() to include stability monitoring.
    """

    @staticmethod
    def forward_with_governance(
        grcm_module,
        x: torch.Tensor,
        want: Optional[torch.Tensor] = None,
        check_qualia: bool = True
    ) -> Dict[str, Any]:
        """
        Modified forward pass with governance integration.

        Args:
            grcm_module: ModularGRCM instance (must have governed_memory attribute)
            x: Input tensor
            want: Optional want signal
            check_qualia: Whether to check qualia conflicts

        Returns:
            Dict with outputs + governance metrics
        """
        batch_size = x.shape[0]

        # Run standard GRCM forward pass
        grounded = grcm_module.grounding(x)
        embedded = grcm_module.embed(grounded)

        # Resonant attention
        attended, coherence = grcm_module.attn(embedded)

        # Desire alignment
        if want is not None:
            want_embedded = grcm_module.embed(want)
            alignment, bandwidth_scale = grcm_module.desire(attended, want_embedded)
        else:
            alignment = torch.ones(batch_size, device=x.device)
            bandwidth_scale = torch.ones_like(alignment)

        # Memory update
        memory_out = grcm_module.memory(
            attended * bandwidth_scale.unsqueeze(-1),
            write_mask=(coherence > grcm_module.config.coherence_threshold).float() * alignment
        )

        # Reflection
        self_state = grcm_module.reflect(memory_out)

        # Qualia
        qualia = grcm_module.qualia(embedded, memory_out)

        # === GOVERNANCE INTEGRATION ===
        if hasattr(grcm_module, 'governed_memory'):
            # Extract GRCM metrics
            grcm_metrics = {
                'coherence': coherence.mean().item(),
                'alignment': alignment.mean().item(),
                'reflection_norm': torch.norm(self_state).item(),
                'qualia_conflicted': qualia['conflicted'].mean().item(),
                'memory_utilization': torch.norm(memory_out).item() / (grcm_module.config.freq_dim ** 0.5)
            }

            # Run governance check
            state_vector = embedded[0].detach().cpu().numpy()
            _, gov_output = grcm_module.governed_memory.process_with_governance(
                state_vector=state_vector,
                grcm_metrics=grcm_metrics,
                torsion=alignment[0].item()
            )

            # Check for brownout
            brownout = grcm_module.governed_memory.should_brownout(gov_output)

            if brownout:
                # System entered brownout - cap outputs
                memory_out = memory_out * gov_output.confidence_cap

        else:
            gov_output = None
            brownout = False

        # Check qualia conflict
        qualia_conflict = check_qualia and (qualia['conflicted'].max() > grcm_module.config.qualia.conflict_threshold)

        # Decode output
        output = grcm_module.decoder(memory_out)

        return {
            'output': output,
            'coherence': coherence,
            'alignment': alignment,
            'qualia': qualia,
            'self_state': self_state,
            'bandwidth_scale': bandwidth_scale,
            'memory_state': memory_out,
            'grounded_input': grounded,
            'qualia_conflict': qualia_conflict,
            'brownout': brownout,
            'gov_output': gov_output,
        }
