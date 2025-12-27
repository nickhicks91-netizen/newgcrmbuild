"""
BentoML Service for GRCM
Production API serving with REST endpoints
"""
import torch
import numpy as np
from typing import Dict, Any, List
import bentoml
from bentoml.io import JSON, NumpyNdarray
import time

from grcm import ModularGRCM, load_config, GRCMOptimizer


# Initialize model
config = load_config("config/grcm_default.yaml")
model = ModularGRCM(config)
model.eval()

# Optional: Load quantized model for production
optimizer = GRCMOptimizer(model)
# quantized_model = optimizer.quantize_dynamic()

# Create BentoML service
svc = bentoml.Service("grcm_consciousness", runners=[])


@svc.api(input=JSON(), output=JSON())
def predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction endpoint

    Input JSON format:
    {
        "image_embedding": [...],  # 512-dim list
        "audio_embedding": [...],  # 768-dim list
        "action": [...],           # 4-dim list (optional)
        "desire_index": 0          # 0-3 (optional)
    }

    Returns:
    {
        "phi": float,
        "coherence": float,
        "qualia": {...},
        "desire_align": float,
        "reflection": float,
        "ethical_status": {...},
        "timestamp": int,
        "latency_ms": float
    }
    """
    start_time = time.time()

    try:
        # Parse inputs
        image_emb = torch.tensor(input_data["image_embedding"], dtype=torch.float32).unsqueeze(0)
        audio_emb = torch.tensor(input_data["audio_embedding"], dtype=torch.float32).unsqueeze(0)

        action = None
        if "action" in input_data:
            action = torch.tensor(input_data["action"], dtype=torch.float32).unsqueeze(0)

        # Set desire if specified
        if "desire_index" in input_data:
            model.set_desire(input_data["desire_index"])

        # Forward pass
        with torch.no_grad():
            outputs = model(image_emb, audio_emb, action)

        # Format response
        response = {
            "phi": float(outputs["phi"]),
            "coherence": float(outputs["coherence"].mean().item()),
            "qualia": {
                "calm": float(outputs["qualia"][0, 0].item()),
                "alert": float(outputs["qualia"][0, 1].item()),
                "curious": float(outputs["qualia"][0, 2].item()),
                "conflicted": float(outputs["qualia"][0, 3].item())
            },
            "desire_align": float(outputs["desire_align"].mean().item()),
            "reflection": float(outputs["reflection"].mean().item()),
            "ethical_status": outputs["ethical_status"],
            "timestamp": outputs["timestamp"],
            "latency_ms": (time.time() - start_time) * 1000
        }

        return response

    except Exception as e:
        return {
            "error": str(e),
            "latency_ms": (time.time() - start_time) * 1000
        }


@svc.api(input=JSON(), output=JSON())
def batch_predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch prediction endpoint

    Input JSON format:
    {
        "image_embeddings": [[...], [...], ...],  # List of 512-dim lists
        "audio_embeddings": [[...], [...], ...],  # List of 768-dim lists
        "actions": [[...], [...], ...],           # List of 4-dim lists (optional)
        "desire_index": 0                         # 0-3 (optional)
    }

    Returns:
    {
        "predictions": [...],  # List of prediction dicts
        "batch_size": int,
        "total_latency_ms": float,
        "avg_latency_per_sample_ms": float
    }
    """
    start_time = time.time()

    try:
        # Parse inputs
        image_embs = torch.tensor(input_data["image_embeddings"], dtype=torch.float32)
        audio_embs = torch.tensor(input_data["audio_embeddings"], dtype=torch.float32)

        batch_size = image_embs.size(0)

        actions = None
        if "actions" in input_data:
            actions = torch.tensor(input_data["actions"], dtype=torch.float32)

        # Set desire if specified
        if "desire_index" in input_data:
            model.set_desire(input_data["desire_index"])

        # Batch forward pass
        with torch.no_grad():
            outputs = model(image_embs, audio_embs, actions)

        # Format batch response
        predictions = []
        for i in range(batch_size):
            pred = {
                "phi": float(outputs["phi"]),  # Scalar for batch
                "coherence": float(outputs["coherence"][i].item()),
                "qualia": {
                    "calm": float(outputs["qualia"][i, 0].item()),
                    "alert": float(outputs["qualia"][i, 1].item()),
                    "curious": float(outputs["qualia"][i, 2].item()),
                    "conflicted": float(outputs["qualia"][i, 3].item())
                },
                "desire_align": float(outputs["desire_align"][i].item()),
                "reflection": float(outputs["reflection"][i].item())
            }
            predictions.append(pred)

        total_latency = (time.time() - start_time) * 1000

        response = {
            "predictions": predictions,
            "batch_size": batch_size,
            "total_latency_ms": total_latency,
            "avg_latency_per_sample_ms": total_latency / batch_size,
            "ethical_status": outputs["ethical_status"],
            "timestamp": outputs["timestamp"]
        }

        return response

    except Exception as e:
        return {
            "error": str(e),
            "latency_ms": (time.time() - start_time) * 1000
        }


@svc.api(input=JSON(), output=JSON())
def get_config() -> Dict[str, Any]:
    """
    Get current model configuration

    Returns:
    {
        "config": {...},
        "version": "0.1.0"
    }
    """
    return {
        "config": model.config.to_dict(),
        "version": "0.1.0",
        "input_dim": model.config.input_dim,
        "freq_dim": model.config.freq_dim,
        "memory_size": model.config.memory_size
    }


@svc.api(input=JSON(), output=JSON())
def set_desire(input_data: Dict[str, int]) -> Dict[str, Any]:
    """
    Set active desire

    Input JSON format:
    {
        "desire_index": 0  # 0-3
    }

    Returns:
    {
        "success": bool,
        "current_desire": int
    }
    """
    try:
        desire_idx = input_data["desire_index"]
        model.set_desire(desire_idx)

        return {
            "success": True,
            "current_desire": model.desire.current_desire_idx
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@svc.api(input=JSON(), output=JSON())
def reset_model(input_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Reset model state

    Returns:
    {
        "success": bool,
        "message": str
    }
    """
    try:
        model.reset()
        return {
            "success": True,
            "message": "Model state reset successfully",
            "timestamp": model.t
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@svc.api(input=JSON(), output=JSON())
def get_state() -> Dict[str, Any]:
    """
    Get full model state for debugging/monitoring

    Returns:
    {
        "state": {...},
        "timestamp": int
    }
    """
    try:
        state = model.get_full_state()
        return {
            "success": True,
            "state": state,
            "timestamp": state["timestamp"]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@svc.api(input=JSON(), output=JSON())
def health() -> Dict[str, str]:
    """
    Health check endpoint

    Returns:
    {
        "status": "healthy",
        "model": "loaded",
        "timestamp": int
    }
    """
    return {
        "status": "healthy",
        "model": "loaded",
        "timestamp": model.t,
        "device": str(next(model.parameters()).device)
    }


# Metrics endpoint for Prometheus
@svc.api(input=JSON(), output=JSON())
def metrics() -> Dict[str, Any]:
    """
    Prometheus-compatible metrics endpoint

    Returns metrics about model state and performance
    """
    state = model.get_full_state()

    return {
        "grcm_timestamp": state["timestamp"],
        "grcm_phi_mean": state["phi"]["mean_phi"],
        "grcm_phi_std": state["phi"]["std_phi"],
        "grcm_memory_norm": state["memory"]["memory_norm"],
        "grcm_memory_updates": state["memory"]["update_count"],
        "grcm_episodes": state["threading"]["num_episodes"],
        "grcm_identity_norm": state["threading"]["identity_norm"]
    }
