"""
Grounding Layer - Multimodal Fusion
Combines CLIP (vision), Wav2Vec (audio), and proprioceptive state
"""
import torch
import torch.nn as nn
from typing import Optional
from ..config import GroundingConfig


class GroundingLayer(nn.Module):
    """
    Multimodal grounding layer that fuses:
    - Visual: CLIP embeddings (512d)
    - Audio: Wav2Vec embeddings (768d)
    - Proprioception: Body state (16d)

    Uses cross-attention for modality harmony.
    """

    def __init__(self, input_dim: int, config: Optional[GroundingConfig] = None):
        super().__init__()
        self.config = config or GroundingConfig()

        # Calculate split dimensions
        self.clip_out_dim = input_dim // 3
        self.wav_out_dim = input_dim // 3
        self.prop_out_dim = input_dim - (self.clip_out_dim + self.wav_out_dim)

        # Projection layers
        self.clip_proj = nn.Linear(self.config.clip_dim, self.clip_out_dim)
        self.wav_proj = nn.Linear(self.config.wav_dim, self.wav_out_dim)
        self.prop_proj = nn.Linear(self.config.prop_dim, self.prop_out_dim)

        # Cross-attention for modality integration
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=self.config.num_heads,
            batch_first=True
        )

        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        image_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        prop_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multimodal inputs into grounded representation

        Args:
            image_emb: CLIP embeddings [batch, 512]
            audio_emb: Wav2Vec embeddings [batch, 768]
            prop_state: Proprioceptive state [batch, 16]

        Returns:
            Grounded representation [batch, input_dim]
        """
        # Project each modality
        clip_proj = self.clip_proj(image_emb)  # [batch, input_dim//3]
        wav_proj = self.wav_proj(audio_emb)    # [batch, input_dim//3]
        prop_proj = self.prop_proj(prop_state) # [batch, remaining]

        # Concatenate modalities
        grounded = torch.cat([clip_proj, wav_proj, prop_proj], dim=-1)  # [batch, input_dim]

        # Add sequence dimension for attention
        grounded_seq = grounded.unsqueeze(1)  # [batch, 1, input_dim]

        # Self-attention for cross-modal integration
        attended, _ = self.cross_attn(grounded_seq, grounded_seq, grounded_seq)

        # Remove sequence dimension and normalize
        output = self.norm(attended.squeeze(1))

        return output

    def get_modality_weights(
        self,
        image_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        prop_state: torch.Tensor
    ) -> dict:
        """
        Compute relative contribution of each modality

        Returns:
            Dictionary with modality weights
        """
        with torch.no_grad():
            clip_norm = self.clip_proj(image_emb).norm(dim=-1)
            wav_norm = self.wav_proj(audio_emb).norm(dim=-1)
            prop_norm = self.prop_proj(prop_state).norm(dim=-1)

            total = clip_norm + wav_norm + prop_norm + 1e-8

            return {
                'vision': (clip_norm / total).mean().item(),
                'audio': (wav_norm / total).mean().item(),
                'proprioception': (prop_norm / total).mean().item()
            }
