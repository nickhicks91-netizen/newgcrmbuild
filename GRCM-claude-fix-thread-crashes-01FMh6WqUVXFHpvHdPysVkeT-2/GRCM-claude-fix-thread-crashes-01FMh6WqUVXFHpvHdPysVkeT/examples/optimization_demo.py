"""
GRCM Optimization Demo
Demonstrates quantization, compilation, and ONNX export
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm import ModularGRCM, load_config, GRCMOptimizer


def main():
    print("=" * 70)
    print("GRCM Optimization Demo")
    print("=" * 70)

    # 1. Load base model
    print("\n[1] Loading base GRCM model...")
    config = load_config("config/grcm_default.yaml")
    model = ModularGRCM(config)
    model.eval()
    print("    ✓ Model loaded")

    # 2. Create optimizer
    print("\n[2] Creating GRCMOptimizer...")
    optimizer = GRCMOptimizer(model, config)
    print("    ✓ Optimizer created")

    # 3. Measure base model size
    print("\n[3] Base model statistics...")
    base_size = optimizer.get_model_size()
    print(f"    Total size: {base_size['total_mb']:.2f} MB")
    print(f"    Parameters: {base_size['params_mb']:.2f} MB")
    print(f"    Buffers: {base_size['buffers_mb']:.2f} MB")

    # 4. Test base model
    print("\n[4] Testing base model forward pass...")
    batch_size = 4
    image_emb = torch.randn(batch_size, 512)
    audio_emb = torch.randn(batch_size, 768)
    action = torch.randn(batch_size, 4)

    with torch.no_grad():
        outputs = model(image_emb, audio_emb, action)

    print(f"    Phi: {outputs['phi']:.3f}")
    print(f"    Coherence: {outputs['coherence'].mean():.3f}")
    print(f"    Qualia: {outputs['qualia'].mean(0).tolist()}")
    print("    ✓ Forward pass successful")

    # 5. Apply dynamic quantization
    print("\n[5] Applying dynamic INT8 quantization...")
    quantized_model = optimizer.quantize_dynamic(dtype=torch.qint8)
    print("    ✓ Quantization complete")

    # 6. Measure quantized model size
    print("\n[6] Quantized model statistics...")
    quant_size = optimizer.get_model_size(quantized_model)
    print(f"    Total size: {quant_size['total_mb']:.2f} MB")
    print(f"    Size reduction: {((base_size['total_mb'] - quant_size['total_mb']) / base_size['total_mb'] * 100):.1f}%")

    # 7. Test quantized model
    print("\n[7] Testing quantized model...")
    with torch.no_grad():
        quant_outputs = quantized_model(image_emb, audio_emb, action)

    print(f"    Phi: {quant_outputs['phi']:.3f}")
    print(f"    Coherence: {quant_outputs['coherence'].mean():.3f}")
    print(f"    Phi difference: {abs(outputs['phi'] - quant_outputs['phi']):.4f}")
    print("    ✓ Quantized model works correctly")

    # 8. torch.compile (if available)
    print("\n[8] Attempting torch.compile()...")
    if hasattr(torch, 'compile'):
        try:
            compiled_model = optimizer.compile_model(backend="inductor", mode=None)
            print("    ✓ Compilation successful")

            # Test compiled model
            with torch.no_grad():
                comp_outputs = compiled_model(image_emb, audio_emb, action)
            print(f"    Phi: {comp_outputs['phi']:.3f}")

        except Exception as e:
            print(f"    ⚠  Compilation failed: {e}")
    else:
        print("    ⚠  torch.compile not available in this PyTorch version")

    # 9. ONNX Export
    print("\n[9] Exporting to ONNX...")
    try:
        # Create output directory
        Path("models").mkdir(exist_ok=True)

        onnx_path = "models/grcm_optimized.onnx"
        metadata = optimizer.export_onnx(
            output_path=onnx_path,
            batch_size=1,
            opset_version=18,
            dynamic_axes=True,
            verbose=False
        )

        print(f"    ✓ ONNX export successful")
        print(f"    Path: {metadata['path']}")
        print(f"    Opset: {metadata['opset_version']}")
        print(f"    Dynamic axes: {metadata['dynamic_axes']}")

        # Check file size
        onnx_size_mb = Path(onnx_path).stat().st_size / (1024 ** 2)
        print(f"    File size: {onnx_size_mb:.2f} MB")

    except Exception as e:
        print(f"    ✗ ONNX export failed: {e}")

    # 10. Size comparison
    print("\n[10] Size Comparison:")
    comparison = optimizer.compare_sizes()
    for model_name, sizes in comparison.items():
        print(f"    {model_name:12s}: {sizes['total_mb']:6.2f} MB")

    print("\n" + "=" * 70)
    print("✓ Optimization demo complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run benchmarks: python examples/benchmark_demo.py")
    print("  - Test ONNX model with ONNXRuntime")
    print("  - Deploy quantized model for production")


if __name__ == "__main__":
    main()
