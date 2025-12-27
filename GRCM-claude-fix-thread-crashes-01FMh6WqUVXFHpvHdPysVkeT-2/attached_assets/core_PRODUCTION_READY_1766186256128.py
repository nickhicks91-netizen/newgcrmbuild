"""
GRCM Core - Main Orchestrator (PRODUCTION-HARDENED)
Integrates all modules for resonant consciousness simulation

CHANGES FROM ORIGINAL:
- Added comprehensive error handling
- Added logging throughout
- Added thread safety with RLock
- Added input validation
- Added graceful degradation
"""
import torch
import torch.nn as nn
import logging
import threading
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

# Set up logging
logger = logging.getLogger(__name__)


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
    
    Thread-safe: Uses RLock for concurrent access
    """

    def __init__(self, config: Optional[GRCMConfig] = None):
        """
        Initialize GRCM with all modules.
        
        Args:
            config: Optional GRCM configuration. Uses defaults if None.
            
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If module initialization fails
        """
        super().__init__()
        
        try:
            self.config = config or GRCMConfig()
            
            # Thread safety: Use RLock (reentrant) to allow nested calls
            self._lock = threading.RLock()
            
            logger.info("Initializing GRCM modules...")
            
            # Initialize all modules with error handling
            try:
                self.grounding = GroundingLayer(
                    self.config.input_dim,
                    self.config.grounding
                )
            except Exception as e:
                logger.error(f"Failed to initialize GroundingLayer: {e}")
                raise RuntimeError(f"GroundingLayer initialization failed: {e}")

            try:
                self.embed = HarmonicEmbedding(
                    self.config.input_dim,
                    self.config.freq_dim
                )
            except Exception as e:
                logger.error(f"Failed to initialize HarmonicEmbedding: {e}")
                raise RuntimeError(f"HarmonicEmbedding initialization failed: {e}")

            try:
                self.attn = ResonantAttention(
                    self.config.freq_dim,
                    self.config.attention
                )
            except Exception as e:
                logger.error(f"Failed to initialize ResonantAttention: {e}")
                raise RuntimeError(f"ResonantAttention initialization failed: {e}")

            try:
                self.desire = DesireModule(
                    self.config.freq_dim,
                    self.config.desire
                )
            except Exception as e:
                logger.error(f"Failed to initialize DesireModule: {e}")
                raise RuntimeError(f"DesireModule initialization failed: {e}")

            try:
                self.memory = MemoryGrid(
                    self.config.memory_size,
                    self.config.freq_dim,
                    self.config.memory
                )
            except Exception as e:
                logger.error(f"Failed to initialize MemoryGrid: {e}")
                raise RuntimeError(f"MemoryGrid initialization failed: {e}")

            self.decoder = nn.Linear(self.config.freq_dim, self.config.memory_size)

            try:
                self.reflect = ReflectionHead(
                    self.config.freq_dim,
                    self.config.memory_size
                )
            except Exception as e:
                logger.error(f"Failed to initialize ReflectionHead: {e}")
                raise RuntimeError(f"ReflectionHead initialization failed: {e}")

            try:
                self.qualia_module = QualiaModule(
                    self.config.freq_dim,
                    self.config.qualia
                )
            except Exception as e:
                logger.error(f"Failed to initialize QualiaModule: {e}")
                raise RuntimeError(f"QualiaModule initialization failed: {e}")

            # Non-module components
            try:
                self.threading = EpisodicThreadBank(
                    self.config.memory_size,
                    self.config.qualia.qualia_dim,
                    self.config.threading
                )
            except Exception as e:
                logger.error(f"Failed to initialize EpisodicThreadBank: {e}")
                raise RuntimeError(f"EpisodicThreadBank initialization failed: {e}")

            try:
                self.phi = PhiEstimator(self.config.phi)
            except Exception as e:
                logger.error(f"Failed to initialize PhiEstimator: {e}")
                raise RuntimeError(f"PhiEstimator initialization failed: {e}")
                
            try:
                self.body = BodySimulator(self.config.body)
            except Exception as e:
                logger.error(f"Failed to initialize BodySimulator: {e}")
                raise RuntimeError(f"BodySimulator initialization failed: {e}")

            # Timestep counter
            self.t = 0
            
            logger.info("GRCM initialized successfully")
            
        except Exception as e:
            logger.error(f"GRCM initialization failed: {e}", exc_info=True)
            raise

    def _validate_inputs(
        self, 
        image_emb: torch.Tensor, 
        audio_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> None:
        """
        Validate forward pass inputs.
        
        Args:
            image_emb: CLIP embeddings
            audio_emb: Wav2Vec embeddings
            action: Optional action vector
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Check None
        if image_emb is None:
            raise ValueError("image_emb cannot be None")
        if audio_emb is None:
            raise ValueError("audio_emb cannot be None")
        
        # Check types
        if not isinstance(image_emb, torch.Tensor):
            raise ValueError(f"image_emb must be torch.Tensor, got {type(image_emb)}")
        if not isinstance(audio_emb, torch.Tensor):
            raise ValueError(f"audio_emb must be torch.Tensor, got {type(audio_emb)}")
        
        # Check shapes
        if image_emb.dim() < 2:
            raise ValueError(f"image_emb must be at least 2D, got shape {image_emb.shape}")
        if audio_emb.dim() < 2:
            raise ValueError(f"audio_emb must be at least 2D, got shape {audio_emb.shape}")
        
        # Check expected dimensions
        if image_emb.shape[-1] != 512:
            raise ValueError(
                f"image_emb last dimension must be 512 (CLIP), got {image_emb.shape[-1]}"
            )
        if audio_emb.shape[-1] != 768:
            raise ValueError(
                f"audio_emb last dimension must be 768 (Wav2Vec), got {audio_emb.shape[-1]}"
            )
        
        # Check batch sizes match
        if image_emb.shape[0] != audio_emb.shape[0]:
            raise ValueError(
                f"Batch size mismatch: image_emb {image_emb.shape[0]} != "
                f"audio_emb {audio_emb.shape[0]}"
            )
        
        # Validate action if provided
        if action is not None:
            if not isinstance(action, torch.Tensor):
                raise ValueError(f"action must be torch.Tensor, got {type(action)}")
            if action.shape[0] != image_emb.shape[0]:
                raise ValueError(
                    f"Action batch size {action.shape[0]} != "
                    f"embedding batch size {image_emb.shape[0]}"
                )

    def forward(
        self,
        image_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Full forward pass with comprehensive error handling.

        Args:
            image_emb: CLIP embeddings [batch, 512]
            audio_emb: Wav2Vec embeddings [batch, 768]
            action: Optional action vector [batch, action_dim]

        Returns:
            Dictionary with all outputs and metrics
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If forward pass fails
        """
        # Thread-safe access
        with self._lock:
            try:
                # Validate inputs
                logger.debug(f"Forward pass t={self.t}, validating inputs...")
                self._validate_inputs(image_emb, audio_emb, action)
                
                logger.debug(
                    f"Input shapes: image={image_emb.shape}, "
                    f"audio={audio_emb.shape}, action={action.shape if action is not None else None}"
                )
                
                self.t += 1

                # 1. Grounding: Fuse multimodal inputs
                try:
                    prop_state = self.body.get_state()
                    grounded = self.grounding(image_emb, audio_emb, prop_state)
                    logger.debug(f"Grounding complete: {grounded.shape}")
                except Exception as e:
                    logger.error(f"Grounding failed: {e}")
                    raise RuntimeError(f"Grounding step failed: {e}")

                # 2. Embedding: Transform to frequency space
                try:
                    identity_token = self.threading.get_identity_token()
                    freq = self.embed(grounded, identity_token)
                    logger.debug(f"Embedding complete: {freq.shape}")
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
                    raise RuntimeError(f"Embedding step failed: {e}")

                # 3. Desire: Compute alignment and bias
                try:
                    desire_align, bw_bias = self.desire(freq)
                    logger.debug(f"Desire computation complete")
                except Exception as e:
                    logger.error(f"Desire computation failed: {e}")
                    raise RuntimeError(f"Desire step failed: {e}")

                # 4. Attention: Compute coherence
                try:
                    coherence = self.attn(freq, bw_bias.mean())
                    logger.debug(f"Coherence: {coherence.mean().item():.4f}")
                except Exception as e:
                    logger.error(f"Attention computation failed: {e}")
                    raise RuntimeError(f"Attention step failed: {e}")

                # 5. Memory: Update with coherent + aligned patterns
                try:
                    desire_mask = self.desire.is_aligned(desire_align)
                    self.memory.update(freq, coherence * desire_mask)
                    mem_read = self.memory()
                    logger.debug(f"Memory update complete")
                except Exception as e:
                    logger.error(f"Memory update failed: {e}")
                    raise RuntimeError(f"Memory step failed: {e}")

                # 6. Reflection: Self-awareness
                try:
                    reflection = self.reflect(freq, mem_read)
                    logger.debug(f"Reflection complete")
                except Exception as e:
                    logger.error(f"Reflection computation failed: {e}")
                    raise RuntimeError(f"Reflection step failed: {e}")

                # 7. Output decoding
                try:
                    out = self.decoder(freq)
                    logger.debug(f"Decoding complete: {out.shape}")
                except Exception as e:
                    logger.error(f"Decoding failed: {e}")
                    raise RuntimeError(f"Decoding step failed: {e}")

                # 8. Qualia: Phenomenal states
                try:
                    qualia = self.qualia_module(freq)
                    logger.debug(f"Qualia computation complete")
                except Exception as e:
                    logger.error(f"Qualia computation failed: {e}")
                    raise RuntimeError(f"Qualia step failed: {e}")

                # 9. Phi: Integrated information
                try:
                    phi_val = self.phi.compute_phi(freq, qualia, mem_read, coherence)
                    logger.debug(f"Phi: {phi_val.mean().item():.4f}")
                except Exception as e:
                    logger.error(f"Phi computation failed: {e}")
                    raise RuntimeError(f"Phi step failed: {e}")

                # 10. Episodic Threading: Store significant episodes
                try:
                    combined_score = (coherence * desire_align).mean()
                    if combined_score > 0.7:
                        self.threading.add_episode(self.t, mem_read, qualia)
                        logger.debug(f"Episode stored (score={combined_score:.4f})")
                except Exception as e:
                    logger.warning(f"Episode storage failed (non-critical): {e}")
                    # Non-critical, continue

                # 11. Body: Update proprioception
                try:
                    if action is not None:
                        prop_state = self.body.update(action, desire_align)
                        logger.debug(f"Body state updated")
                except Exception as e:
                    logger.warning(f"Body update failed (non-critical): {e}")
                    # Non-critical, continue

                # 12. Ethical check
                ethical_status = {}
                try:
                    if self.config.enable_ethical_halt:
                        ethical_status = self.qualia_module.check_ethical_halt(qualia)
                        if ethical_status.get('halted', False):
                            logger.warning(
                                f"Ethical halt triggered: {ethical_status.get('reason')}"
                            )
                except Exception as e:
                    logger.error(f"Ethical check failed: {e}")
                    # Set safe default
                    ethical_status = {'halted': True, 'reason': f'Error in check: {e}'}

                # Assemble output dictionary
                result = {
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
                
                logger.debug(f"Forward pass complete (t={self.t})")
                return result
                
            except ValueError as e:
                # Input validation errors - re-raise with context
                logger.error(f"Input validation failed at t={self.t}: {e}")
                raise
            except RuntimeError as e:
                # Processing errors - re-raise with context
                logger.error(f"Forward pass failed at t={self.t}: {e}")
                raise
            except Exception as e:
                # Unexpected errors - wrap and raise
                logger.error(
                    f"Unexpected error in forward pass at t={self.t}: {e}", 
                    exc_info=True
                )
                raise RuntimeError(f"Unexpected forward pass error: {e}")

    def set_desire(self, desire_idx: int) -> None:
        """
        Set active desire.
        
        Args:
            desire_idx: Index of desire to activate
            
        Raises:
            ValueError: If desire_idx is invalid
        """
        with self._lock:
            try:
                if not isinstance(desire_idx, int):
                    raise ValueError(f"desire_idx must be int, got {type(desire_idx)}")
                
                self.desire.set_desire(desire_idx)
                logger.info(f"Desire set to {desire_idx}")
                
            except Exception as e:
                logger.error(f"Failed to set desire: {e}")
                raise

    def reset(self) -> None:
        """Reset all stateful components (thread-safe)."""
        with self._lock:
            try:
                logger.info("Resetting GRCM state...")
                
                self.memory.reset()
                self.threading.reset()
                self.phi.reset()
                self.body.reset()
                self.t = 0
                
                logger.info("GRCM reset complete")
                
            except Exception as e:
                logger.error(f"Reset failed: {e}", exc_info=True)
                raise RuntimeError(f"Reset failed: {e}")

    def get_full_state(self) -> Dict[str, Any]:
        """
        Get comprehensive system state for analysis (thread-safe).

        Returns:
            Dictionary with all module states
            
        Raises:
            RuntimeError: If state retrieval fails
        """
        with self._lock:
            try:
                logger.debug("Retrieving full system state...")
                
                state = {
                    'timestamp': self.t,
                    'memory': self.memory.get_memory_stats(),
                    'threading': self.threading.get_threading_stats(),
                    'phi': self.phi.get_phi_stats(),
                    'body': self.body.get_body_stats(),
                    'config': self.config.to_dict()
                }
                
                logger.debug("State retrieval complete")
                return state
                
            except Exception as e:
                logger.error(f"Failed to get state: {e}", exc_info=True)
                raise RuntimeError(f"State retrieval failed: {e}")

    @classmethod
    def from_config_file(cls, config_path: str) -> "ModularGRCM":
        """
        Create GRCM from YAML config file.

        Args:
            config_path: Path to YAML config

        Returns:
            ModularGRCM instance
            
        Raises:
            ValueError: If config file is invalid
            FileNotFoundError: If config file doesn't exist
        """
        try:
            logger.info(f"Loading config from {config_path}")
            config = load_config(config_path)
            return cls(config)
            
        except FileNotFoundError as e:
            logger.error(f"Config file not found: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load config: {e}", exc_info=True)
            raise ValueError(f"Invalid config file: {e}")

    def save_config(self, save_path: str) -> None:
        """
        Save current config to YAML.
        
        Args:
            save_path: Path to save config
            
        Raises:
            IOError: If save fails
        """
        try:
            logger.info(f"Saving config to {save_path}")
            self.config.to_yaml(save_path)
            logger.info("Config saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}", exc_info=True)
            raise IOError(f"Config save failed: {e}")
