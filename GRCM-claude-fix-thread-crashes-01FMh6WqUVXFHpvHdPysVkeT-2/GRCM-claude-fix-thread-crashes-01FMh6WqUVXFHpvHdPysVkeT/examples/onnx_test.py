"""
GRCM ONNX Export and Testing
Export model to ONNX and optionally test with ONNXRuntime
"""
import torch
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm import ModularGRCM, load_config


def export_onnx():
    """Export GRCM to ONNX format"""
    print("=" * 70)
    print("GRCM ONNX Export")
    print("=" * 70)

    # Load model
    print("\n[1] Loading GRCM model...")
    config = load_config("config/grcm_default.yaml")
    model = ModularGRCM(config)
    model.eval()
    print("    ✓ Model loaded")

    # Prepare dummy inputs
    print("\n[2] Preparing dummy inputs...")
    batch_size = 1
    image_emb = torch.randn(batch_size, 512)
    audio_emb = torch.randn(batch_size, 768)
    action = torch.randn(batch_size, 4)
    print(f"    Batch size: {batch_size}")

    # Test PyTorch forward pass
    print("\n[3] Testing PyTorch forward pass...")
    with torch.no_grad():
        pt_outputs = model(image_emb, audio_emb, action)

    print(f"    Phi: {pt_outputs['phi']:.3f}")
    print(f"    Coherence: {pt_outputs['coherence'].mean():.3f}")
    print(f"    Qualia: {pt_outputs['qualia'][0].tolist()}")

    # Export to ONNX
    print("\n[4] Exporting to ONNX...")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "grcm_model.onnx"

    # Wrapper to convert dict output to tuple
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image_emb, audio_emb, action):
            outputs = self.model(image_emb, audio_emb, action)
            # Convert dict to tuple for ONNX
            return (
                outputs['output'],
                outputs['coherence'],
                outputs['memory'].unsqueeze(0),  # Add batch dim
                outputs['reflection'],
                outputs['qualia'],
                outputs['identity_token'].unsqueeze(0),  # Add batch dim
                outputs['desire_align'],
                torch.tensor([outputs['phi']]),  # Convert scalar to tensor
                outputs['prop_state']
            )

    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()

    # Define dynamic axes
    dynamic_axes = {
        'image_embeddings': {0: 'batch'},
        'audio_embeddings': {0: 'batch'},
        'action': {0: 'batch'},
        'output': {0: 'batch'},
        'coherence': {0: 'batch'},
        'qualia': {0: 'batch'},
        'desire_align': {0: 'batch'},
        'prop_state': {0: 'batch'},
    }

    # Export
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (image_emb, audio_emb, action),
            str(onnx_path),
            input_names=['image_embeddings', 'audio_embeddings', 'action'],
            output_names=[
                'output', 'coherence', 'memory', 'reflection',
                'qualia', 'identity_token', 'desire_align',
                'phi', 'prop_state'
            ],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
            verbose=False
        )

    file_size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"    ✓ Export successful!")
    print(f"    Path: {onnx_path}")
    print(f"    File size: {file_size_mb:.2f} MB")
    print(f"    Opset version: 18")
    print(f"    Dynamic axes: Enabled")

    return str(onnx_path), pt_outputs


def test_onnx_runtime(onnx_path: str, pt_outputs: dict):
    """Test ONNX model with ONNXRuntime"""
    print("\n[5] Testing with ONNXRuntime...")

    try:
        import onnxruntime as ort
    except ImportError:
        print("    ⚠  ONNXRuntime not installed")
        print("    Install with: pip install onnxruntime")
        return

    # Create session
    print("    Loading ONNX model...")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print("    ✓ Model loaded in ONNXRuntime")

    # Prepare inputs
    batch_size = 1
    inputs = {
        'image_embeddings': np.random.randn(batch_size, 512).astype(np.float32),
        'audio_embeddings': np.random.randn(batch_size, 768).astype(np.float32),
        'action': np.random.randn(batch_size, 4).astype(np.float32)
    }

    # Run inference
    print("    Running inference...")
    outputs = sess.run(None, inputs)

    # Print outputs
    print("    ✓ Inference successful!")
    print(f"    Output shapes:")
    for i, name in enumerate(sess.get_outputs()):
        print(f"      {name.name}: {outputs[i].shape}")

    # Extract key values
    qualia_onnx = outputs[4]
    phi_onnx = outputs[7][0]

    print(f"\n    ONNX Results:")
    print(f"      Phi: {phi_onnx:.3f}")
    print(f"      Qualia: {qualia_onnx[0].tolist()}")

    print("\n    ✓ ONNXRuntime test complete!")


def main():
    # Export
    onnx_path, pt_outputs = export_onnx()

    # Test with ONNXRuntime if available
    test_onnx_runtime(onnx_path, pt_outputs)

    print("\n" + "=" * 70)
    print("✓ ONNX export complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Test ONNX model with different batch sizes")
    print("  - Deploy with ONNX Runtime Server")
    print("  - Convert to TensorRT for GPU inference")
    print("  - Benchmark ONNX vs PyTorch performance")


if __name__ == "__main__":
    main()
