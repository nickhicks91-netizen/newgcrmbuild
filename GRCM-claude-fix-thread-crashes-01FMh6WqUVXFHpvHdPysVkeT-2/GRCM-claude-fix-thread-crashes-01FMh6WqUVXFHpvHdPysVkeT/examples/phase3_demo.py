"""
Phase 3 Demo: Testing, Logging, and Visualization
Demonstrates pytest, MLflow logging, and Gradio UI
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from grcm import (
    ModularGRCM,
    load_config,
    MLflowLogger,
    GRCMExperiment,
    GRCMInterface
)


def demo_mlflow_logging():
    """Demonstrate MLflow logging"""
    print("=" * 70)
    print("MLflow Logging Demo")
    print("=" * 70)

    # Load model
    print("\n[1] Loading GRCM model...")
    config = load_config("config/grcm_default.yaml")
    model = ModularGRCM(config)
    print("    ✓ Model loaded")

    # Create logger
    print("\n[2] Creating MLflow logger...")
    logger = MLflowLogger(
        experiment_name="GRCM-Phase3-Demo",
        enable=True
    )

    if not logger.mlflow_available:
        print("    ⚠  MLflow not available")
        print("    Install with: pip install mlflow")
        return

    print("    ✓ Logger created")

    # Start run
    print("\n[3] Starting MLflow run...")
    with logger:
        # Log configuration
        logger.log_config(config)
        print("    ✓ Configuration logged")

        # Run forward passes
        print("\n[4] Running 10 forward passes...")
        for step in range(10):
            image_emb = torch.randn(4, 512)
            audio_emb = torch.randn(4, 768)
            action = torch.randn(4, 4)

            outputs = model(image_emb, audio_emb, action)

            # Log outputs
            logger.log_model_outputs(outputs, step)

            if step % 3 == 0:
                print(f"    Step {step}: Phi={outputs['phi']:.3f}, Coherence={outputs['coherence'].mean():.3f}")

        print("    ✓ Forward passes complete")

        # Log final state
        print("\n[5] Logging final model state...")
        state = model.get_full_state()

        logger.log_metrics({
            'final_phi_mean': state['phi']['mean_phi'],
            'final_coherence': state['memory']['avg_coherence'],
            'total_steps': state['timestamp']
        })
        print("    ✓ Final state logged")

    print("\n[6] MLflow run complete!")
    print(f"    Run ID: {logger.run_id}")
    print("\n    View results with:")
    print("      mlflow ui")
    print("      # Then open http://localhost:5000")


def demo_experiment_tracking():
    """Demonstrate high-level experiment tracking"""
    print("\n" + "=" * 70)
    print("Experiment Tracking Demo")
    print("=" * 70)

    print("\n[1] Creating experiment...")
    config = load_config("config/grcm_default.yaml")
    model = ModularGRCM(config)

    experiment = GRCMExperiment(
        model,
        experiment_name="GRCM-Forward-Experiment",
        enable_logging=True
    )
    print("    ✓ Experiment created")

    if not experiment.logger.mlflow_available:
        print("    ⚠  MLflow not available, skipping")
        return

    print("\n[2] Running experiment...")
    experiment.run_forward_experiment(
        num_steps=20,
        batch_size=4,
        log_interval=5
    )
    print("    ✓ Experiment complete")


def demo_gradio_ui():
    """Demonstrate Gradio UI"""
    print("\n" + "=" * 70)
    print("Gradio UI Demo")
    print("=" * 70)

    try:
        from grcm.ui import create_gradio_ui
    except ImportError:
        print("\n    ⚠  Gradio not available")
        print("    Install with: pip install gradio")
        return

    print("\n[1] Creating Gradio UI...")
    demo = create_gradio_ui("config/grcm_default.yaml")
    print("    ✓ UI created")

    print("\n[2] UI features:")
    print("    - Desire selection (slider)")
    print("    - Action control (X, Y sliders)")
    print("    - Noise level adjustment")
    print("    - Real-time qualia visualization")
    print("    - Phi trajectory plotting")
    print("    - Coherence monitoring")
    print("    - Ethical halt detection")

    print("\n[3] To launch UI:")
    print("    >>> from grcm.ui import launch_ui")
    print("    >>> launch_ui()")
    print("    # Then open http://localhost:7860")


def demo_testing_overview():
    """Overview of testing capabilities"""
    print("\n" + "=" * 70)
    print("Testing Overview")
    print("=" * 70)

    print("\n[Test Suite Structure]")
    print("  tests/")
    print("  ├── conftest.py           # Shared fixtures")
    print("  ├── test_core.py          # Core GRCM tests (25+ tests)")
    print("  ├── test_modules.py       # Module unit tests (50+ tests)")
    print("  └── test_integration.py   # Integration tests (20+ tests)")

    print("\n[Test Coverage]")
    print("  - Unit tests: All 10 modules")
    print("  - Integration tests: Full pipeline")
    print("  - Stress tests: Large batches, many iterations")
    print("  - Configuration tests: Various configs")

    print("\n[Running Tests]")
    print("  # Run all tests")
    print("  pytest tests/")
    print()
    print("  # Run with coverage")
    print("  pytest tests/ --cov=grcm --cov-report=html")
    print()
    print("  # Run specific markers")
    print("  pytest tests/ -m unit          # Unit tests only")
    print("  pytest tests/ -m integration   # Integration tests")
    print("  pytest tests/ -m slow          # Slow tests")
    print("  pytest tests/ -m stress        # Stress tests")

    print("\n[Test Markers]")
    print("  @pytest.mark.unit          # Fast unit tests")
    print("  @pytest.mark.integration   # Integration tests")
    print("  @pytest.mark.slow          # Tests >1 second")
    print("  @pytest.mark.stress        # Stress tests (large scale)")
    print("  @pytest.mark.gpu           # GPU-required tests")


def demo_metrics_tracking():
    """Demonstrate metrics tracked by GRCM"""
    print("\n" + "=" * 70)
    print("Metrics Tracking")
    print("=" * 70)

    print("\n[1] Forward Pass Metrics:")
    print("  - Phi (Φ): Integrated information")
    print("  - Coherence: Resonance level")
    print("  - Qualia: [calm, alert, curious, conflicted]")
    print("  - Desire Alignment: Goal alignment")
    print("  - Reflection: Self-awareness")
    print("  - Memory norm: Accumulated experience")
    print("  - Identity token: Narrative continuity")

    print("\n[2] Training Metrics:")
    print("  - Loss: MSE(align, labels) - Phi")
    print("  - Phi trajectory")
    print("  - Alignment error")
    print("  - Loss improvement")

    print("\n[3] Performance Metrics:")
    print("  - Latency (ms): Forward pass time")
    print("  - Throughput (samples/sec)")
    print("  - Memory usage (MB)")
    print("  - Coherence distribution")
    print("  - Phi stability")

    print("\n[4] Ethical Metrics:")
    print("  - Conflict ratio: qualia[conflicted] > 0.6")
    print("  - Coherence compliance: >95% above 0.7")
    print("  - Phi awareness: >1.5 threshold")
    print("  - Halt triggers: Safety events")


def main():
    print("\n" + "=" * 70)
    print("GRCM Phase 3: Testing, Logging & Visualization")
    print("=" * 70)

    # 1. Testing overview
    demo_testing_overview()

    # 2. MLflow logging
    try:
        demo_mlflow_logging()
    except Exception as e:
        print(f"\n    ⚠  MLflow demo failed: {e}")

    # 3. Experiment tracking
    try:
        demo_experiment_tracking()
    except Exception as e:
        print(f"\n    ⚠  Experiment demo failed: {e}")

    # 4. Gradio UI
    demo_gradio_ui()

    # 5. Metrics overview
    demo_metrics_tracking()

    # Summary
    print("\n" + "=" * 70)
    print("Phase 3 Summary")
    print("=" * 70)

    print("\n✅ Testing:")
    print("  - 95+ unit and integration tests")
    print("  - Pytest configuration with markers")
    print("  - Fixtures for models, configs, and data")
    print("  - Stress tests for scalability")

    print("\n✅ Logging:")
    print("  - MLflow integration")
    print("  - Experiment tracking")
    print("  - Metrics and artifacts logging")
    print("  - Model versioning")

    print("\n✅ Visualization:")
    print("  - Interactive Gradio UI")
    print("  - Real-time qualia visualization")
    print("  - Phi and coherence plotting")
    print("  - Ethical halt monitoring")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)

    print("\n1. Run tests:")
    print("   pytest tests/ -v")

    print("\n2. View coverage:")
    print("   pytest tests/ --cov=grcm --cov-report=html")
    print("   open htmlcov/index.html")

    print("\n3. Start MLflow UI:")
    print("   mlflow ui")
    print("   open http://localhost:5000")

    print("\n4. Launch Gradio UI:")
    print("   python -c \"from grcm.ui import launch_ui; launch_ui()\"")
    print("   open http://localhost:7860")

    print("\n5. Run experiments:")
    print("   python examples/phase3_demo.py")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
