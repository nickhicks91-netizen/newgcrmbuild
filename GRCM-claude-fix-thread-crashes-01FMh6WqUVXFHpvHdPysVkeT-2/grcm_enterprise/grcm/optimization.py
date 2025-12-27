"""
GRCM Optimization Utilities
Quantization, compilation, and export for production deployment
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import warnings

from .core import ModularGRCM
from .config import GRCMConfig


class GRCMOptimizer:
    """
    Optimization wrapper for GRCM model

    Supports:
    - Dynamic INT8 quantization
    - torch.compile() integration
    - ONNX export
    - Performance benchmarking
    """

    def __init__(self, model: ModularGRCM, config: Optional[GRCMConfig] = None):
        self.model = model
        self.config = config or model.config
        self.optimized_model = None
        self.quantized_model = None
        self.compiled_model = None

    def quantize_dynamic(
        self,
        dtype: torch.dtype = torch.qint8,
        modules_to_quantize: Optional[set] = None
    ) -> nn.Module:
        """
        Apply dynamic INT8 quantization

        Args:
            dtype: Quantization dtype (qint8 or float16)
            modules_to_quantize: Specific module types to quantize

        Returns:
            Quantized model
        """
        if modules_to_quantize is None:
            modules_to_quantize = {nn.Linear, nn.MultiheadAttention, nn.GRUCell}

        print(f"[Quantization] Applying dynamic quantization with {dtype}...")

        # Quantize the model
        quantized = torch.quantization.quantize_dynamic(
            self.model,
            modules_to_quantize,
            dtype=dtype
        )

        self.quantized_model = quantized
        print(f"[Quantization] ✓ Complete")

        return quantized

    def compile_model(
        self,
        backend: str = "inductor",
        mode: Optional[str] = None,
        fullgraph: bool = False
    ) -> nn.Module:
        """
        Compile model with torch.compile()

        Args:
            backend: Compilation backend ('inductor', 'aot_eager', 'cudagraphs')
            mode: Optimization mode (None, 'reduce-overhead', 'max-autotune')
            fullgraph: Require full graph compilation

        Returns:
            Compiled model
        """
        if not hasattr(torch, 'compile'):
            warnings.warn("torch.compile not available in this PyTorch version")
            return self.model

        print(f"[Compilation] Compiling with backend={backend}, mode={mode}...")

        try:
            compiled = torch.compile(
                self.model,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph
            )
            self.compiled_model = compiled
            print(f"[Compilation] ✓ Complete")
            return compiled
        except Exception as e:
            warnings.warn(f"Compilation failed: {e}")
            return self.model

    def export_onnx(
        self,
        output_path: str,
        batch_size: int = 1,
        opset_version: int = 18,
        dynamic_axes: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Export model to ONNX format

        Args:
            output_path: Path to save ONNX model
            batch_size: Example batch size for export
            opset_version: ONNX opset version
            dynamic_axes: Enable dynamic batch/sequence axes
            verbose: Verbose export logging

        Returns:
            Export metadata
        """
        print(f"[ONNX Export] Exporting to {output_path}...")

        # Prepare dummy inputs
        dummy_image = torch.randn(batch_size, 512)
        dummy_audio = torch.randn(batch_size, 768)
        dummy_action = torch.randn(batch_size, 4)

        # Define input names
        input_names = ['image_embeddings', 'audio_embeddings', 'action']

        # Define output names
        output_names = [
            'output', 'coherence', 'memory', 'reflection',
            'qualia', 'identity_token', 'desire_align',
            'phi', 'prop_state'
        ]

        # Define dynamic axes if enabled
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'image_embeddings': {0: 'batch'},
                'audio_embeddings': {0: 'batch'},
                'action': {0: 'batch'},
                'output': {0: 'batch'},
                'coherence': {0: 'batch'},
                'qualia': {0: 'batch'},
                'desire_align': {0: 'batch'},
                'prop_state': {0: 'batch'},
            }

        # Wrapper to convert dict output to tuple
        class ONNXWrapper(nn.Module):
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

        wrapped_model = ONNXWrapper(self.model)
        wrapped_model.eval()

        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapped_model,
                    (dummy_image, dummy_audio, dummy_action),
                    output_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes_dict,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    verbose=verbose
                )

            print(f"[ONNX Export] ✓ Complete: {output_path}")

            # Return metadata
            metadata = {
                'path': output_path,
                'opset_version': opset_version,
                'batch_size': batch_size,
                'dynamic_axes': dynamic_axes,
                'input_names': input_names,
                'output_names': output_names
            }

            return metadata

        except Exception as e:
            print(f"[ONNX Export] ✗ Failed: {e}")
            raise

    def optimize_all(
        self,
        quantize: bool = True,
        compile: bool = False,
        export_onnx: bool = True,
        onnx_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply all optimizations

        Args:
            quantize: Apply quantization
            compile: Apply torch.compile
            export_onnx: Export to ONNX
            onnx_path: Path for ONNX export

        Returns:
            Optimization results
        """
        results = {}

        if quantize:
            results['quantized_model'] = self.quantize_dynamic()

        if compile:
            results['compiled_model'] = self.compile_model()

        if export_onnx:
            if onnx_path is None:
                onnx_path = "grcm_model.onnx"
            results['onnx_metadata'] = self.export_onnx(onnx_path)

        return results

    def get_model_size(self, model: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Calculate model size in MB

        Args:
            model: Model to measure (default: self.model)

        Returns:
            Size metrics
        """
        if model is None:
            model = self.model

        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size

        return {
            'total_mb': total_size / (1024 ** 2),
            'params_mb': param_size / (1024 ** 2),
            'buffers_mb': buffer_size / (1024 ** 2)
        }

    def compare_sizes(self) -> Dict[str, Dict[str, float]]:
        """
        Compare sizes of original vs optimized models

        Returns:
            Size comparison dictionary
        """
        comparison = {
            'original': self.get_model_size(self.model)
        }

        if self.quantized_model is not None:
            comparison['quantized'] = self.get_model_size(self.quantized_model)

        return comparison


def create_optimized_model(
    config_path: str = "config/grcm_default.yaml",
    quantize: bool = True,
    compile: bool = False,
    export_onnx: bool = True,
    onnx_output: str = "models/grcm_optimized.onnx"
) -> Tuple[ModularGRCM, GRCMOptimizer, Dict[str, Any]]:
    """
    Convenience function to create and optimize GRCM model

    Args:
        config_path: Path to config YAML
        quantize: Apply quantization
        compile: Apply torch.compile
        export_onnx: Export to ONNX
        onnx_output: ONNX export path

    Returns:
        (original_model, optimizer, results)
    """
    from .core import ModularGRCM
    from .config import load_config

    # Load model
    config = load_config(config_path)
    model = ModularGRCM(config)

    # Create optimizer
    optimizer = GRCMOptimizer(model, config)

    # Apply optimizations
    results = optimizer.optimize_all(
        quantize=quantize,
        compile=compile,
        export_onnx=export_onnx,
        onnx_path=onnx_output
    )

    # Add size comparison
    results['size_comparison'] = optimizer.compare_sizes()

    return model, optimizer, results
