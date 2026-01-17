"""
Utility functions for GRCM integrations
"""
import torch
from typing import Dict, Any


QUALIA_LABELS = ['calm', 'alert', 'curious', 'conflicted']


def qualia_tensor_to_dict(qualia: torch.Tensor) -> Dict[str, float]:
    """
    Convert qualia tensor to dictionary format
    
    Args:
        qualia: Qualia tensor [batch, 4] or [4]
        
    Returns:
        Dictionary with qualia labels as keys
    """
    if isinstance(qualia, dict):
        return qualia
        
    if qualia.dim() > 1:
        qualia = qualia.squeeze(0)
        
    return {
        label: float(qualia[i]) 
        for i, label in enumerate(QUALIA_LABELS)
    }


def normalize_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize GRCM outputs for integration use
    
    Converts tensors to Python types where appropriate
    
    Args:
        outputs: Raw outputs from ModularGRCM.forward()
        
    Returns:
        Normalized outputs dict
    """
    normalized = dict(outputs)
    
    if 'qualia' in normalized and isinstance(normalized['qualia'], torch.Tensor):
        normalized['qualia'] = qualia_tensor_to_dict(normalized['qualia'])
        
    return normalized
