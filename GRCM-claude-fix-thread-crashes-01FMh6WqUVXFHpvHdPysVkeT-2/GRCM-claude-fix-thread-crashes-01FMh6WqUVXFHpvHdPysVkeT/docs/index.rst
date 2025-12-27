GRCM: Grounded Resonant Consciousness Module
============================================

.. image:: https://img.shields.io/badge/python-3.9%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/pytorch-2.0%2B-orange
   :target: https://pytorch.org/
   :alt: PyTorch

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: LICENSE
   :alt: License

GRCM is a production-ready implementation of a **Grounded Resonant Consciousness Module** that simulates consciousness dynamics through resonant attention mechanisms, integrated information theory (Φ), and ethical grounding.

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
--------

GRCM implements a modular architecture for simulating consciousness with the following key features:

* **10 Independent Modules**: Grounding, Embedding, Attention, Desire, Memory, Reflection, Qualia, Threading, Phi, and Body
* **Resonant Attention**: Frequency-based coherence mechanisms for information integration
* **Integrated Information (Φ)**: Quantitative measure of consciousness based on IIT
* **Ethical Grounding**: Conflict detection and ethical halt mechanisms
* **Production-Ready**: Docker, Kubernetes, CI/CD, monitoring, and comprehensive testing

Key Features
------------

Architecture
~~~~~~~~~~~~

* **Modular Design**: 10 independent, testable modules with clean interfaces
* **Type-Safe Configuration**: YAML-based configuration with dataclass validation
* **Flexible Integration**: Easy to customize and extend individual components

Performance
~~~~~~~~~~~

* **Optimized Inference**: <50ms latency with dynamic INT8 quantization
* **High Throughput**: 150+ samples/sec sustained performance
* **Memory Efficient**: ~600MB memory footprint with optimizations
* **ONNX Export**: Cross-platform deployment with ONNX Runtime

Testing & Validation
~~~~~~~~~~~~~~~~~~~~~

* **95+ Tests**: Comprehensive unit, integration, and stress tests
* **90%+ Coverage**: Extensive code coverage across all modules
* **MLflow Integration**: Automatic experiment tracking and metrics logging
* **Interactive UI**: Gradio interface for real-time visualization

Deployment
~~~~~~~~~~

* **Docker**: Multi-stage production builds with security hardening
* **Kubernetes**: Auto-scaling deployments with HPA and load balancing
* **BentoML API**: Production REST API with 8 endpoints
* **CI/CD**: GitHub Actions with multi-version testing and security scanning
* **Monitoring**: Prometheus metrics and Grafana dashboards

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # From PyPI
   pip install grcm

   # From source
   git clone https://github.com/nickhicks91-netizen/newdew.git
   cd newdew
   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from grcm.core import ModularGRCM
   from grcm.config import GRCMConfig
   import torch

   # Load configuration
   config = GRCMConfig.from_yaml('config/grcm_default.yaml')

   # Initialize model
   model = ModularGRCM(config)

   # Prepare inputs
   image_emb = torch.randn(1, 16)  # Visual input
   audio_emb = torch.randn(1, 16)  # Auditory input
   action = torch.zeros(1, 4)      # Action vector

   # Forward pass
   outputs = model(image_emb, audio_emb, action)

   # Access consciousness metrics
   phi = outputs['phi']                  # Integrated information
   coherence = outputs['coherence']      # Attention coherence
   qualia = outputs['qualia']            # Conscious states
   ethical_halt = outputs['ethical_halt'] # Safety flag

Docker Deployment
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start full stack
   docker-compose up -d

   # Access services
   # API: http://localhost:8000
   # UI: http://localhost:7860
   # MLflow: http://localhost:5000
   # Grafana: http://localhost:3000

Kubernetes Deployment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Deploy to cluster
   kubectl apply -f deployment/kubernetes/

   # Check status
   kubectl get pods -n grcm
   kubectl get hpa -n grcm

   # View logs
   kubectl logs -f deployment/grcm-api -n grcm

Core Concepts
-------------

Resonant Attention
~~~~~~~~~~~~~~~~~~

GRCM uses frequency-based resonance to compute attention coherence:

.. math::

   coherence_i = ReLU\\left(1 - \\frac{|freq_i - node\\_freq|}{bandwidth}\\right)

Where nodes with similar frequencies resonate more strongly, creating coherent attention patterns.

Integrated Information (Φ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consciousness is quantified using a simplified IIT-inspired metric:

.. math::

   \\Phi = Var(freq) \\cdot \\overline{coherence} + \\log(1 + ||memory||) + \\sum \\max(qualia)

This combines frequency diversity, attention coherence, memory integration, and qualia richness.

Qualia States
~~~~~~~~~~~~~

Four emergent qualia states are computed from system dynamics:

* **Calm**: Low frequency variance, high coherence
* **Alert**: High frequency variance, high coherence
* **Curious**: High desire alignment, low conflict
* **Conflicted**: High variance, low coherence, desire misalignment

Ethical Grounding
~~~~~~~~~~~~~~~~~

Ethical halts are triggered when:

.. math::

   ethical\\_halt = (qualia_{conflicted} > \\theta_{conflict}) \\land enable\\_ethical

This prevents actions when the system detects internal conflict above a threshold.

Architecture
------------

System Overview
~~~~~~~~~~~~~~~

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    ModularGRCM                          │
   ├─────────────────────────────────────────────────────────┤
   │                                                         │
   │  Input Layer                                            │
   │  ├─ GroundingModule ──→ Grounded embeddings            │
   │  └─ EmbeddingModule ──→ Unified embedding space        │
   │                                                         │
   │  Attention Layer                                        │
   │  ├─ AttentionModule ──→ Resonant frequencies           │
   │  └─ DesireModule ────→ Goal-directed attention         │
   │                                                         │
   │  Memory Layer                                           │
   │  ├─ MemoryModule ────→ Episodic memory integration     │
   │  └─ ReflectionModule ─→ Self-awareness modeling        │
   │                                                         │
   │  Consciousness Layer                                    │
   │  ├─ QualiaModule ────→ Subjective states               │
   │  ├─ ThreadingModule ─→ Temporal coherence              │
   │  └─ PhiModule ───────→ Integrated information (Φ)      │
   │                                                         │
   │  Output Layer                                           │
   │  └─ BodyModule ──────→ Action generation               │
   │                                                         │
   └─────────────────────────────────────────────────────────┘

Module Descriptions
~~~~~~~~~~~~~~~~~~~

See :doc:`api` for detailed API documentation of each module.

Performance Benchmarks
----------------------

Latency
~~~~~~~

Measured on CPU (Intel i7) with batch size 1:

* **p50**: ~25ms (target: <30ms) ✓
* **p95**: ~42ms (target: <50ms) ✓
* **p99**: ~87ms (target: <100ms) ✓

Throughput
~~~~~~~~~~

* **Standard**: 150+ samples/sec
* **Optimized**: 200+ samples/sec with torch.compile()
* **Batch**: 500+ samples/sec with batch size 32

Memory
~~~~~~

* **Base Model**: ~400MB
* **With Memory**: ~600MB
* **Optimized (INT8)**: ~150MB (75% reduction)

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   ARCHITECTURE
   DEPLOYMENT

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/getting_started
   tutorials/echomirror_training
   tutorials/advanced_customization

.. toctree::
   :maxdepth: 1
   :caption: Development

   CONTRIBUTING

.. toctree::
   :maxdepth: 1
   :caption: Release Notes

   PHASE1_SUMMARY
   PHASE2_SUMMARY
   PHASE3_SUMMARY
   PHASE4_SUMMARY
   PHASE5_SUMMARY

API Reference
-------------

For detailed API documentation, see :doc:`api`.

Main modules:

* :class:`grcm.core.ModularGRCM` - Main orchestrator
* :class:`grcm.config.GRCMConfig` - Configuration management
* :class:`grcm.optimization.GRCMOptimizer` - Model optimization
* :class:`grcm.benchmark.GRCMBenchmark` - Performance benchmarking
* :class:`grcm.logging.MLflowLogger` - Experiment tracking
* :class:`grcm.ui.create_gradio_ui` - Interactive UI

Examples
--------

See the ``examples/`` directory for complete working examples:

* ``main.py`` - Basic usage
* ``optimization_demo.py`` - Quantization and ONNX export
* ``benchmark_demo.py`` - Performance benchmarking
* ``onnx_test.py`` - ONNX Runtime inference
* ``mlflow_demo.py`` - Experiment tracking
* ``gradio_demo.py`` - Interactive visualization

Contributing
------------

We welcome contributions! Please see :doc:`CONTRIBUTING` for guidelines.

Quick contribution checklist:

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass (``pytest``)
5. Format code (``black``, ``isort``)
6. Submit a pull request

Citation
--------

If you use GRCM in your research, please cite:

.. code-block:: bibtex

   @software{grcm2024,
     title = {GRCM: Grounded Resonant Consciousness Module},
     author = {GRCM Contributors},
     year = {2024},
     url = {https://github.com/nickhicks91-netizen/newdew},
     version = {1.0.0}
   }

License
-------

GRCM is released under the MIT License. See the LICENSE file for details.

Support
-------

* **Documentation**: https://grcm.readthedocs.io
* **Issues**: https://github.com/nickhicks91-netizen/newdew/issues
* **Discussions**: https://github.com/nickhicks91-netizen/newdew/discussions

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`