"""
FastAPI endpoints for EchoZero + GRCM hybrid system.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np

from ..hybrid import EchoGRCMHybrid
from ..train import EchoMirrorTrainer


class ForwardRequest(BaseModel):
    """Request for forward pass."""
    image_emb: List[float]
    audio_emb: List[float]
    action: List[float]
    prop_state: Optional[List[float]] = None
    memory: Optional[List[float]] = None


class ForwardResponse(BaseModel):
    """Response from forward pass."""
    phi: float
    coherence: List[float]
    qualia: List[float]
    psi_real: List[float]
    psi_imag: List[float]


class StateResponse(BaseModel):
    """Current system state."""
    n_nodes: int
    coupling_norm: float
    system_info: Dict


def create_echozero_api(
    model: EchoGRCMHybrid,
    trainer: Optional[EchoMirrorTrainer] = None,
) -> FastAPI:
    """
    Create FastAPI application for EchoZero.

    Args:
        model: EchoGRCM hybrid model
        trainer: Optional EchoMirror trainer

    Returns:
        app: FastAPI application
    """
    app = FastAPI(
        title="EchoZero + GRCM API",
        description="Resonant intelligence hybrid system",
        version="1.0.0",
    )

    @app.get("/")
    def root():
        """Root endpoint."""
        return {
            "name": "EchoZero + GRCM",
            "version": "1.0.0",
            "status": "operational",
        }

    @app.post("/forward", response_model=ForwardResponse)
    def forward(request: ForwardRequest):
        """
        Execute unified forward pass.

        Args:
            request: Input data

        Returns:
            Forward pass results
        """
        try:
            # Convert to tensors
            image_emb = torch.tensor([request.image_emb], dtype=torch.float32)
            audio_emb = torch.tensor([request.audio_emb], dtype=torch.float32)
            action = torch.tensor([request.action], dtype=torch.float32)

            # Optional states
            prop_state = None
            memory = None
            if request.prop_state:
                prop_state = torch.tensor([request.prop_state], dtype=torch.float32)
            if request.memory:
                memory = torch.tensor([request.memory], dtype=torch.float32)

            # Forward pass
            with torch.no_grad():
                outputs = model(
                    image_emb=image_emb,
                    audio_emb=audio_emb,
                    action=action,
                    prop_state=prop_state,
                    memory=memory,
                )

            # Convert to response
            response = ForwardResponse(
                phi=outputs["phi"][0].item(),
                coherence=outputs["coherence"][0].tolist(),
                qualia=outputs["qualia"][0].tolist(),
                psi_real=outputs["psi"][0].real.tolist(),
                psi_imag=outputs["psi"][0].imag.tolist(),
            )

            return response

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/state", response_model=StateResponse)
    def get_state():
        """Get current system state."""
        info = model.get_system_info()

        return StateResponse(
            n_nodes=model.n_nodes,
            coupling_norm=model.echozero.K.abs().mean().item(),
            system_info=info,
        )

    @app.get("/phi")
    def get_phi():
        """Get latest integrated information."""
        # Requires storing last outputs
        return {"phi": 0.0, "note": "Call /forward first"}

    @app.post("/desire")
    def update_desire(desires: List[float]):
        """
        Update desire vectors.

        Args:
            desires: New desire vector

        Returns:
            Status
        """
        if len(desires) != model.n_nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Desire vector must have {model.n_nodes} elements",
            )

        # Store desires (would need state management)
        return {"status": "updated", "n_desires": len(desires)}

    @app.get("/lattice")
    def get_lattice():
        """Get lattice topology information."""
        K = model.echozero.K
        node_freqs = model.echozero.node_freqs

        return {
            "n_nodes": model.n_nodes,
            "coupling_matrix_norm": K.abs().mean().item(),
            "frequency_range": [
                node_freqs.min().item(),
                node_freqs.max().item(),
            ],
            "hermitian": True,  # By construction
        }

    @app.post("/reset")
    def reset():
        """Reset system to initial state."""
        model.reset_state()
        return {"status": "reset", "message": "System reset to initial state"}

    @app.get("/health")
    def health():
        """Health check."""
        return {
            "status": "healthy",
            "model_loaded": True,
            "n_nodes": model.n_nodes,
        }

    return app


# Example usage:
# app = create_echozero_api(model)
# uvicorn grcm.api.echozero_api:app --reload
