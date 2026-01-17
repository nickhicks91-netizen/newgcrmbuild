"""
GRCM Configuration Management
Dataclasses for type-safe configuration with YAML support
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class AttentionConfig:
    """Resonant Attention Configuration"""
    freq_dim: int = 8
    base_bandwidth: float = 0.5
    coherence_threshold: float = 0.7
    bandwidth_min: float = 0.1
    bandwidth_max: float = 1.0


@dataclass
class MemoryConfig:
    """Memory Grid Configuration"""
    memory_size: int = 32
    update_threshold: float = 0.7


@dataclass
class DesireConfig:
    """Desire Module Configuration"""
    num_desires: int = 4
    alignment_threshold: float = 0.5
    bandwidth_bias_scale: float = 0.2
    default_desire_idx: int = 0


@dataclass
class ThreadingConfig:
    """Episodic Threading Configuration"""
    max_episodes: int = 50
    min_episodes_for_arc: int = 2
    arc_scale: float = 0.1


@dataclass
class PhiConfig:
    """Phi Estimation Configuration"""
    awareness_threshold: float = 1.5
    history_max_size: int = 1000


@dataclass
class BodyConfig:
    """Body Simulator Configuration"""
    state_dim: int = 16
    dt: float = 0.1
    mass: float = 1.0


@dataclass
class GroundingConfig:
    """Multimodal Grounding Configuration"""
    clip_dim: int = 512
    wav_dim: int = 768
    prop_dim: int = 16
    num_heads: int = 3


@dataclass
class QualiaConfig:
    """Qualia Configuration"""
    qualia_dim: int = 4
    qualia_labels: list = field(default_factory=lambda: ["calm", "alert", "curious", "conflicted"])
    conflict_threshold: float = 0.6


@dataclass
class TrainingConfig:
    """EchoMirror Training Configuration"""
    learning_rate: float = 0.01
    num_epochs: int = 5
    phi_loss_weight: float = 1.0


@dataclass
class GRCMConfig:
    """Master GRCM Configuration"""
    # Model dimensions
    input_dim: int = 15
    freq_dim: int = 8
    memory_size: int = 32

    # Submodule configs
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    desire: DesireConfig = field(default_factory=DesireConfig)
    threading: ThreadingConfig = field(default_factory=ThreadingConfig)
    phi: PhiConfig = field(default_factory=PhiConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    qualia: QualiaConfig = field(default_factory=QualiaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Runtime settings
    device: str = "cpu"
    dtype: str = "float32"
    enable_ethical_halt: bool = True
    enable_logging: bool = False
    log_interval: int = 10

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GRCMConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GRCMConfig":
        """Create config from dictionary"""
        # Extract nested configs
        attention_cfg = AttentionConfig(**config_dict.get('attention', {}))
        memory_cfg = MemoryConfig(**config_dict.get('memory', {}))
        desire_cfg = DesireConfig(**config_dict.get('desire', {}))
        threading_cfg = ThreadingConfig(**config_dict.get('threading', {}))
        phi_cfg = PhiConfig(**config_dict.get('phi', {}))
        body_cfg = BodyConfig(**config_dict.get('body', {}))
        grounding_cfg = GroundingConfig(**config_dict.get('grounding', {}))
        qualia_cfg = QualiaConfig(**config_dict.get('qualia', {}))
        training_cfg = TrainingConfig(**config_dict.get('training', {}))

        # Remove nested dicts and create main config
        main_config = {k: v for k, v in config_dict.items()
                      if k not in ['attention', 'memory', 'desire', 'threading',
                                   'phi', 'body', 'grounding', 'qualia', 'training']}

        return cls(
            **main_config,
            attention=attention_cfg,
            memory=memory_cfg,
            desire=desire_cfg,
            threading=threading_cfg,
            phi=phi_cfg,
            body=body_cfg,
            grounding=grounding_cfg,
            qualia=qualia_cfg,
            training=training_cfg
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = asdict(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


def load_config(config_path: Optional[str] = None) -> GRCMConfig:
    """
    Load GRCM configuration from file or return default

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        GRCMConfig instance
    """
    if config_path is None:
        return GRCMConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return GRCMConfig.from_yaml(str(path))
