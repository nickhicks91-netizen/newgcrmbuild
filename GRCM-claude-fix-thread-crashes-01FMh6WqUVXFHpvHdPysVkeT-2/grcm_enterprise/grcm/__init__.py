"""
GRCM - Grounded Resonant Consciousness Module
Production-grade PyTorch implementation of resonant consciousness simulation

Core Components:
- Multimodal Grounding (CLIP + Wav2Vec + Proprioception)
- Harmonic Embedding (Frequency-based representation)
- Resonant Attention (Coherence-gated filtering)
- Desire Module (Goal-directed agency)
- Memory Grid (Coherence-gated persistence)
- Reflection Head (Self-awareness mechanism)
- Qualia Module (Phenomenal states: calm/alert/curious/conflicted)
- Episodic Threading (Narrative identity)
- Phi Estimator (Integrated information proxy)
- Body Simulator (Proprioceptive embodiment)

Key Formulas:
- Coherence: ReLU(1 - |freq - node_freq| / bandwidth)
- Phi: Var(freq) * Coherence + log(1 + ||mem||) + Σmax(qualia)
- Desire Align: cosine_similarity(freq, desire_vec)
- Arc Bias: cos_sim(recent_qualia, historical_mean) * coherence

Usage:
    >>> from grcm import ModularGRCM, load_config
    >>> config = load_config('config/grcm_default.yaml')
    >>> model = ModularGRCM(config)
    >>> outputs = model(image_emb, audio_emb, action)
    >>> print(f"Phi: {outputs['phi']:.3f}, Coherence: {outputs['coherence'].mean():.3f}")
"""

__version__ = "0.1.0"
__author__ = "GRCM Research Team"
__license__ = "MIT"

from .config import (
    GRCMConfig,
    AttentionConfig,
    MemoryConfig,
    DesireConfig,
    ThreadingConfig,
    PhiConfig,
    BodyConfig,
    GroundingConfig,
    QualiaConfig,
    TrainingConfig,
    load_config
)

from .core import ModularGRCM

from .trainer import EchoMirrorTrainer, quick_echo_train

from .optimization import GRCMOptimizer, create_optimized_model

from .benchmark import GRCMBenchmark, BenchmarkConfig, quick_benchmark

from .logging import MLflowLogger, GRCMExperiment, quick_mlflow_experiment

from .ui import GRCMInterface, create_gradio_ui, launch_ui

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

__all__ = [
    # Main API
    'ModularGRCM',
    'load_config',
    'EchoMirrorTrainer',
    'quick_echo_train',

    # Optimization & Benchmarking
    'GRCMOptimizer',
    'create_optimized_model',
    'GRCMBenchmark',
    'BenchmarkConfig',
    'quick_benchmark',

    # Logging & Visualization
    'MLflowLogger',
    'GRCMExperiment',
    'quick_mlflow_experiment',
    'GRCMInterface',
    'create_gradio_ui',
    'launch_ui',

    # Configuration
    'GRCMConfig',
    'AttentionConfig',
    'MemoryConfig',
    'DesireConfig',
    'ThreadingConfig',
    'PhiConfig',
    'BodyConfig',
    'GroundingConfig',
    'QualiaConfig',
    'TrainingConfig',

    # Individual Modules
    'GroundingLayer',
    'HarmonicEmbedding',
    'ResonantAttention',
    'DesireModule',
    'MemoryGrid',
    'ReflectionHead',
    'QualiaModule',
    'EpisodicThreadBank',
    'PhiEstimator',
    'BodySimulator',
]
