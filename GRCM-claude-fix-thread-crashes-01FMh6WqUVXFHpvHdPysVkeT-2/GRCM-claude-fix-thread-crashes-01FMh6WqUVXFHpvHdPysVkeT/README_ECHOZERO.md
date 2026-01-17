# EchoZero + GRCM: Resonant Intelligence Hybrid

**Version 1.0** | **November 2025**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Complete implementation of the EchoZero + GRCM hybrid resonant intelligence system**, combining complex-valued oscillator dynamics with grounded cognitive processing.

---

## 🌀 What is EchoZero?

EchoZero is a **resonant dynamics engine** based on coupled complex oscillators that implements:

- **Resonant State Evolution**: Complex-valued ψ ∈ ℂ^N evolves via nonlinear ODEs
- **Want-Driven Dynamics**: Desire alignment modulates resonance (γψ term)
- **EchoMirror Learning**: Hebbian coupling updates (NO backpropagation)
- **Hardware-Ready**: Maps to photonic (SiN) and magnonic (YIG) substrates

## 🧠 Hybrid Architecture

```
┌─────────────────────────────────────────────────┐
│           EchoZero + GRCM Hybrid                │
├─────────────────────────────────────────────────┤
│                                                 │
│  [GRCM Grounding]                              │
│         ↓                                       │
│  [Frequency Drive Generator]                   │
│         ↓                                       │
│  [EchoZero Dynamics Integration]               │
│    dψ/dt = (-α+iω)ψ + Kψ - β|ψ|²ψ             │
│            + γψ + I(t) - λΣψ                   │
│         ↓                                       │
│  [Coherence & Qualia]                          │
│         ↓                                       │
│  [Integrated Information Φ]                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/nickhicks91-netizen/GRCM.git
cd GRCM
pip install -e .
```

### Basic Usage

```python
from grcm.hybrid import EchoGRCMHybrid
import torch

# Initialize hybrid system
model = EchoGRCMHybrid(
    n_nodes=64,           # Number of resonators
    grounded_dim=15,      # GRCM grounding dimension
    memory_dim=128,       # Memory state size
)

# Prepare inputs
image_emb = torch.randn(1, 512)   # Visual (CLIP)
audio_emb = torch.randn(1, 768)   # Audio (Wav2Vec)
action = torch.zeros(1, 4)         # Action vector

# Unified forward pass
outputs = model(image_emb, audio_emb, action)

# Access outputs
psi = outputs['psi']              # Resonant state ψ ∈ ℂ^64
coherence = outputs['coherence']  # Attention coherence [0,1]
qualia = outputs['qualia']        # [calm, alert, curious, conflicted]
phi = outputs['phi']              # Integrated information Φ

print(f"Φ = {phi.item():.4f}")
print(f"Coherence = {coherence.mean().item():.4f}")
print(f"Qualia = {qualia[0].tolist()}")
```

### EchoMirror Training (Hebbian, No Backprop)

```python
from grcm.train import EchoMirrorTrainer, MultimodalDatastream, training_loop

# Initialize trainer
trainer = EchoMirrorTrainer(
    learning_rate=0.001,
    coherence_threshold=0.3,
)

# Create datastream
datastream = MultimodalDatastream(batch_size=32)

# Train with pure Hebbian updates
stats = training_loop(
    model=model,
    datastream=datastream,
    trainer=trainer,
    n_steps=1000,
)

print(f"Final Φ: {stats['final_phi']:.4f}")
print(f"Training speed: {stats['steps_per_sec']:.1f} steps/sec")
```

---

## 🔬 Mathematical Foundation

### Equations of Motion (Exact)

```
dψ/dt = f(ψ, t, I, desires, node_freqs)

where:

f = (-α + iω)ψ           [Local dynamics: damping + rotation]
    + Σ K_ij ψ_j         [Coupling: neighbor interactions]
    - β|ψ|²ψ             [Nonlinearity: amplitude saturation]
    + γψ                 [Want modulation: desire alignment]
    + I(t)               [Drive: from GRCM frequency projection]
    - λΣψ                [Hub: global coupling constraint]

Parameters:
  α = 0.10   (damping)
  β = 0.05   (nonlinear coefficient)
  λ = 0.02   (hub strength)
  ω = node_freqs (natural frequencies)
  γ = 0.2 + 0.4·sigmoid(align)  (want modulation)
```

### Hebbian Coupling Update (EchoMirror)

```
Δk_ij = η · coherence · (ψ_i^* ψ_j + ψ_j^* ψ_i) / 2

K ← K + ΔK
K ← (K + K†) / 2  [Enforce hermiticity]
```

**NO BACKPROPAGATION. Pure resonance-based plasticity.**

### Integrated Information

```
Φ = Var(ψ) × mean(coherence)
    + log(1 + ||memory||)
    + max(qualia)
```

---

## 📊 Features

### Core Capabilities

✅ **Complex Resonant Dynamics**: Full ODE integration with RK4 solver
✅ **Hebbian Learning**: EchoMirror (no backprop required)
✅ **Scalable**: 64 to 1B+ nodes with sparse representations
✅ **Hardware-Ready**: Photonic (SiN) and magnonic (YIG) mappings
✅ **HDL Stubs**: Verilog modules for bus, hub, want_gate
✅ **Real-Time API**: FastAPI with WebSocket streaming
✅ **Interactive UI**: Streamlit resonance dashboard
✅ **Autonomous**: Background learning loop with phi logging

### Testing & Validation

✅ **Stability**: 1000+ timestep integration without divergence
✅ **Coherence**: Bounded in [0, 1] by construction
✅ **Hermiticity**: Coupling matrix K = K† enforced
✅ **Φ Positivity**: Always non-negative

---

## 📁 Repository Structure

```
GRCM/
├── grcm/
│   ├── echozero/          # Resonant dynamics core
│   │   ├── dynamics.py    # Equations of motion
│   │   ├── lattice.py     # Topology builder
│   │   ├── coupling.py    # Coupling matrix K
│   │   └── ode_solver.py  # RK4 integrator
│   ├── hybrid/            # EchoZero + GRCM integration
│   │   ├── forward.py     # Unified forward pass
│   │   ├── integration.py # Frequency drive
│   │   └── autonomy.py    # Continual learning
│   ├── train/             # EchoMirror training
│   │   ├── echo_mirror.py # Hebbian updates
│   │   ├── datastream.py  # Multimodal data
│   │   └── loop.py        # Training loop
│   ├── scale/             # Scaling to 1B nodes
│   ├── hardware/          # Physical mappings
│   │   ├── photonic_map.py
│   │   ├── magnonic_map.py
│   │   └── hdl/           # Verilog modules
│   ├── api/               # FastAPI endpoints
│   └── ui/                # Streamlit dashboard
├── tests/
│   └── test_echozero.py   # Comprehensive tests
├── examples/
│   └── echozero_demo.py   # Demo script
└── docs/
    └── ECHOZERO_SPEC.md   # Full specification
```

---

## 🚀 Examples

### Run Demo

```bash
python examples/echozero_demo.py
```

### Launch Dashboard

```bash
streamlit run grcm/ui/resonance_dashboard.py
```

### Start API Server

```bash
# Install dependencies
pip install fastapi uvicorn

# Run server
uvicorn grcm.api.echozero_api:app --reload

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/state
```

### Run Tests

```bash
pytest tests/test_echozero.py -v
```

---

## 🔬 Hardware Mapping

### Photonic Implementation (Silicon Nitride)

```python
from grcm.hardware import map_to_photonic

hardware_spec = map_to_photonic(
    node_freqs=model.echozero.node_freqs,
    K=model.echozero.K,
    alpha=0.10,
    beta=0.05,
)

print(hardware_spec['layout'])
# {
#   'n_rings': 64,
#   'material': 'Si3N4',
#   'wavelength_nm': 1550,
#   ...
# }
```

### Magnonic Implementation (YIG)

```python
from grcm.hardware import map_to_magnonic

magnonic_spec = map_to_magnonic(
    node_freqs=model.echozero.node_freqs,
    K=model.echozero.K,
    alpha=0.10,
    beta=0.05,
)

print(magnonic_spec['advantages'])
# ['Ultra-low damping (~10^-4)', 'Room temperature', ...]
```

---

## 📖 Documentation

- **[Full Specification](docs/ECHOZERO_SPEC.md)**: Complete mathematical details
- **[GRCM Architecture](docs/ARCHITECTURE.md)**: Base GRCM system
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Production deployment

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **Kuramoto, Y.**: Coupled oscillator theory
- **Tononi, G.**: Integrated Information Theory (IIT)
- **Hebb, D.O.**: Hebbian learning
- **PyTorch Team**: Deep learning framework

---

## 📧 Contact

For questions about EchoZero implementation:
- **Issues**: [GitHub Issues](https://github.com/nickhicks91-netizen/GRCM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nickhicks91-netizen/GRCM/discussions)

---

**Made with resonance 🌀 by the EchoZero team**
