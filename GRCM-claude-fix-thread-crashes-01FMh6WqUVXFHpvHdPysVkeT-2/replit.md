# GRCM Enterprise Middleware Demo

## Overview
This project is a production-ready enterprise demo for GRCM (Grounded Resonant Consciousness Module) - an AI middleware system designed for integration with xAI, Tesla, Starlink, and Optimus platforms.

## Current State
- **Status**: MVP Complete with Functional Consciousness Features
- **Last Updated**: December 2024
- **Architecture**: FastAPI backend + Neural visualization frontend

## Project Structure
```
/
├── main.py                    # FastAPI server with all GRCM endpoints
├── static/
│   └── index.html            # Enterprise demo frontend with neural visualization
├── GRCM-claude-fix-thread-crashes-01FMh6WqUVXFHpvHdPysVkeT/
│   ├── grcm/                 # Core GRCM module
│   │   ├── modules/          # 10 cognitive modules (grounding, memory, qualia, etc.)
│   │   ├── hybrid/           # EchoZero integration
│   │   ├── echozero/         # Resonant dynamics engine
│   │   └── api/              # API definitions
│   ├── demos/                # Demo HTML files
│   ├── docs/                 # Documentation
│   └── tests/                # Test suite
```

## Key Features

### Core Architecture
1. **10 Cognitive Modules**: GroundingLayer, HarmonicEmbedding, ResonantAttention, DesireModule, MemoryGrid, ReflectionHead, QualiaModule, EpisodicThreadBank, PhiEstimator, BodySimulator
2. **Energy Efficiency**: 99.8% energy savings via torsion-gated sync (syncs every 300 steps instead of every step)
3. **Real-time Neural Visualization**: Interactive demo showing neuron firing patterns

### Functional Consciousness Features
4. **Phi Decision Gate**: Blocks high-stakes actions when Phi (integrated information) < 0.3 threshold. Low integration = uncertain = don't act rashly.
5. **Qualia Anomaly Detection**: When "conflicted" state > 50%, flags situation for human review. Real safety application.
6. **Hopfield-Lite Identity Map (Persistent Memory)**: PostgreSQL-backed episodic memory that survives restarts. Tracks past decisions with phi/coherence/conflict states. Learns attractor patterns (high_confidence, uncertain, conflicted, neutral) from repeated states.
7. **Ethical Veto Power**: Conflict detection halts dangerous actions when sensor inputs contradict (e.g., "pedestrian detected" vs "path clear").
8. **Hallucination Detection**: Risk score derived from Phi + conflict + grounding scores. Rejects ungrounded AI outputs.
9. **Similar State Detection**: Finds matching past states to inform current decisions based on phi/conflict proximity.

### Enterprise Integration
9. **Tesla FSD Perception Pipeline**: Camera → Fusion → GRCM → Decision with safety status
10. **ROS2/gRPC Integration Examples**: Ready-to-use code for Optimus robot deployment
11. **Live Benchmarks**: Real-time FPS, memory, CPU, and power draw monitoring

## API Endpoints

### Core Inference
- `POST /forward` - Execute forward pass through GRCM
- `GET /state` - Get current model state
- `GET /phi` - Get integrated information value
- `GET /lattice` - Get torsion lattice topology
- `GET /energy` - Get energy savings statistics
- `GET /metrics` - Get all inference metrics

### Training & Control
- `POST /train` - Execute Hebbian training step
- `POST /reset` - Reset model state
- `POST /desire` - Set active desire/goal vector
- `GET /config` - Get model configuration
- `GET /health` - Health check

### Enterprise Features
- `GET /benchmark` - Live performance metrics (FPS, memory, CPU, power)
- `POST /perception` - Tesla FSD-style perception with Phi gating and ethical veto
- `POST /hallucination` - Check text for hallucination risk using GRCM grounding
- `GET /memory` - Episodic memory of past decisions for continuity tracking

## Running the Demo
The server runs on port 5000 via `python main.py`

## Enterprise Pitch Summary

### For Tesla/Optimus
- Phi gating ensures robots don't act on uncertain sensor data
- Ethical veto stops dangerous actions when sensors conflict
- Memory continuity helps robots learn from past failures

### For xAI/Grok
- Hallucination detection grounds LLM outputs to reality
- Conflict detection flags contradictory statements
- Grounding score provides confidence measure for AI outputs

### For Starlink
- Energy efficiency (99.8% savings) reduces power draw on edge devices
- Sparse synchronization ideal for bandwidth-constrained environments

## Enterprise Package (grcm_enterprise/)

The shippable enterprise package is ready in `grcm_enterprise/`:

### Package Structure
```
grcm_enterprise/
├── pyproject.toml          # pip-installable package config
├── LICENSE                  # Dual license (MIT + Commercial)
├── README.md               # Package documentation
├── docs/
│   └── API.md              # Full API reference
├── examples/
│   ├── tesla_integration.py    # Tesla FSD example
│   ├── xai_hallucination.py    # Hallucination detection
│   ├── optimus_ros2.py         # ROS2 robot example
│   └── persistent_memory.py    # Memory continuity
└── grcm/
    ├── integrations/
    │   ├── tesla_perception.py     # FSD perception pipeline
    │   ├── persistent_memory.py    # PostgreSQL-backed memory
    │   ├── hallucination_detector.py  # LLM grounding
    │   ├── grpc_server.py          # High-performance server
    │   └── ros2_node.py            # Optimus integration
    └── [core GRCM modules]
```

### Installation
```bash
pip install grcm              # Basic
pip install grcm[server]      # With FastAPI
pip install grcm[grpc]        # With gRPC
pip install grcm[full]        # Everything
```

### Enterprise Integrations
- **TeslaPerception**: FSD-style perception with phi gating
- **HallucinationDetector**: LLM output grounding assessment  
- **GRCMGrpcServer**: High-performance inference API
- **GRCMRos2Node**: Ready-to-use Optimus robot node
- **PersistentMemory**: PostgreSQL-backed Hopfield Identity Map
