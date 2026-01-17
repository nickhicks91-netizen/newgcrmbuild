"""
GRCM Core - Main Orchestrator
Integrates all modules for resonant consciousness simulation
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .config import GRCMConfig, load_config
from .modules import (
    GroundingLayer,
    HarmonicEmbedding,
    ResonantAttention,
    DesireModule,
    MemoryGrid,
    ReflectionHead,
    QualiaModule,
    EpisodicThreadBank,
    PhiEstimator,
    BodySimulator
)


class ModularGRCM(nn.Module):
    """
    Modular Grounded Resonant Consciousness Module

    Architecture:
    1. Grounding: Fuse vision (CLIP) + audio (Wav2Vec) + proprioception
    2. Embedding: Transform to frequency space with memory modulation
    3. Attention: Compute resonance coherence
    4. Desire: Compute alignment and bandwidth bias
    5. Memory: Update with coherent + aligned patterns
    6. Reflection: Self-awareness through memory projection
    7. Qualia: Phenomenal states (calm/alert/curious/conflicted)
    8. Threading: Episodic memory and identity evolution
    9. Phi: Integrated information measure
    10. Body: Proprioceptive feedback from actions

    Ethical safeguard: Halts if qualia[conflicted] > threshold
    """

    def __init__(self, config: Optional[GRCMConfig] = None):
        super().__init__()

        self.config = config or GRCMConfig()

        # Initialize all modules
        self.grounding = GroundingLayer(
            self.config.input_dim,
            self.config.grounding
        )

        self.embed = HarmonicEmbedding(
            self.config.input_dim,
            self.config.freq_dim
        )

        self.attn = ResonantAttention(
            self.config.freq_dim,
            self.config.attention
        )

        self.desire = DesireModule(
            self.config.freq_dim,
            self.config.desire
        )

        self.memory = MemoryGrid(
            self.config.memory_size,
            self.config.freq_dim,
            self.config.memory
        )

        self.decoder = nn.Linear(self.config.freq_dim, self.config.memory_size)

        self.reflect = ReflectionHead(
            self.config.freq_dim,
            self.config.memory_size
        )

        self.qualia_module = QualiaModule(
            self.config.freq_dim,
            self.config.qualia
        )

        # Non-module components
        self.threading = EpisodicThreadBank(
            self.config.memory_size,
            self.config.qualia.qualia_dim,
            self.config.threading
        )

        self.phi = PhiEstimator(self.config.phi)
        self.body = BodySimulator(self.config.body)

        # Timestep counter
        self.t = 0

    def forward(
        self,
        image_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Full forward pass

        Args:
            image_emb: CLIP embeddings [batch, 512]
            audio_emb: Wav2Vec embeddings [batch, 768]
            action: Optional action vector [batch, action_dim]

        Returns:
            Dictionary with all outputs and metrics
        """
        self.t += 1

        # 1. Grounding: Fuse multimodal inputs
        prop_state = self.body.get_state()
        grounded = self.grounding(image_emb, audio_emb, prop_state)

        # 2. Embedding: Transform to frequency space
        identity_token = self.threading.get_identity_token()
        freq = self.embed(grounded, identity_token)

        # 3. Desire: Compute alignment and bias
        desire_align, bw_bias = self.desire(freq)

        # 4. Attention: Compute coherence
        coherence = self.attn(freq, bw_bias.mean())

        # 5. Memory: Update with coherent + aligned patterns
        desire_mask = self.desire.is_aligned(desire_align)
        self.memory.update(freq, coherence * desire_mask)
        mem_read = self.memory()

        # 6. Reflection: Self-awareness
        reflection = self.reflect(freq, mem_read)

        # 7. Output decoding
        out = self.decoder(freq)

        # 8. Qualia: Phenomenal states
        qualia = self.qualia_module(freq)

        # 9. Phi: Integrated information
        phi_val = self.phi.compute_phi(freq, qualia, mem_read, coherence)

        # 10. Episodic Threading: Store significant episodes
        combined_score = (coherence * desire_align).mean()
        if combined_score > 0.7:
            self.threading.add_episode(self.t, mem_read, qualia)

        # 11. Body: Update proprioception
        if action is not None:
            prop_state = self.body.update(action, desire_align)

        # 12. Ethical check
        ethical_status = {}
        if self.config.enable_ethical_halt:
            ethical_status = self.qualia_module.check_ethical_halt(qualia)

        # Assemble output dictionary
        return {
            'output': out,
            'coherence': coherence,
            'memory': mem_read,
            'reflection': reflection,
            'qualia': qualia,
            'identity_token': identity_token,
            'desire_align': desire_align,
            'phi': phi_val,
            'prop_state': prop_state,
            'ethical_status': ethical_status,
            'timestamp': self.t
        }

    def set_desire(self, desire_idx: int) -> None:
        """Set active desire"""
        self.desire.set_desire(desire_idx)

    def reset(self) -> None:
        """Reset all stateful components"""
        self.memory.reset()
        self.threading.reset()
        self.phi.reset()
        self.body.reset()
        self.t = 0

    def get_full_state(self) -> Dict[str, Any]:
        """
        Get comprehensive system state for analysis

        Returns:
            Dictionary with all module states
        """
        return {
            'timestamp': self.t,
            'memory': self.memory.get_memory_stats(),
            'threading': self.threading.get_threading_stats(),
            'phi': self.phi.get_phi_stats(),
            'body': self.body.get_body_stats(),
            'config': self.config.to_dict()
        }

    @classmethod
    def from_config_file(cls, config_path: str) -> "ModularGRCM":
        """
        Create GRCM from YAML config file

        Args:
            config_path: Path to YAML config

        Returns:
            ModularGRCM instance
        """
        config = load_config(config_path)
        return cls(config)

    def save_config(self, save_path: str) -> None:
        """Save current config to YAML"""
        self.config.to_yaml(save_path)
