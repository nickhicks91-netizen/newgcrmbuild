"""
EchoZero Dynamics - Core equations of motion (PRODUCTION-HARDENED).

Implements the exact resonant dynamics from the canonical specification:
    dψ/dt = local + coupled + nonlinear + want + drive + hub

NO backpropagation. Pure forward dynamics.

CHANGES FROM ORIGINAL:
- Added comprehensive error handling
- Added logging throughout
- Added input validation
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class EchoZeroSystem(nn.Module):
    """
    Complete EchoZero resonant dynamics system.

    State Variables:
        ψ ∈ ℂ^N: Resonant state (complex-valued)
        K ∈ ℂ^(N×N): Coupling matrix
        node_freqs ∈ ℝ^N: Natural frequencies

    Parameters:
        α = 0.10: Damping coefficient
        β = 0.05: Nonlinear damping
        λ = 0.02: Hub coupling strength
    """

    def __init__(
        self,
        n_nodes: int,
        coupling_matrix: torch.Tensor,
        node_freqs: torch.Tensor,
        alpha: float = 0.10,
        beta: float = 0.05,
        lambda_hub: float = 0.02,
        device: str = "cpu",
    ):
        """
        Initialize EchoZero system with error handling.

        Args:
            n_nodes: Number of oscillator nodes
            coupling_matrix: K ∈ ℂ^(N×N) coupling matrix
            node_freqs: Natural frequencies ω ∈ ℝ^N
            alpha: Damping coefficient
            beta: Nonlinear damping
            lambda_hub: Hub constraint strength
            device: Computation device
        """
        super().__init__()
        
        try:
            logger.info(f"Initializing EchoZeroSystem with {n_nodes} nodes")
            
            self.n_nodes = n_nodes
            self.alpha = alpha
            self.beta = beta
            self.lambda_hub = lambda_hub
            self.device = device

            self.register_buffer("K", coupling_matrix.to(device))
            self.register_buffer("node_freqs", node_freqs.to(device))

            self.psi = None
            self.reset()
            
            logger.info("EchoZeroSystem initialized successfully")
        except Exception as e:
            logger.error(f"EchoZeroSystem initialization failed: {e}")
            raise

    def reset(self):
        """Reset to random initial state with error handling."""
        try:
            real = torch.randn(self.n_nodes, device=self.device) * 0.1
            imag = torch.randn(self.n_nodes, device=self.device) * 0.1
            self.psi = torch.complex(real, imag)
            logger.debug("EchoZeroSystem state reset")
        except Exception as e:
            logger.error(f"EchoZeroSystem reset failed: {e}")
            raise

    def forward(
        self,
        psi: torch.Tensor,
        I_t: torch.Tensor,
        desires: torch.Tensor,
        t: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute dψ/dt according to EchoZero dynamics with error handling.

        Args:
            psi: Current state ψ ∈ ℂ^N
            I_t: Drive signal I(t) ∈ ℂ^N
            desires: Desire vectors ∈ ℝ^N
            t: Time (optional, for time-dependent drive)

        Returns:
            dpsi_dt: Time derivative dψ/dt ∈ ℂ^N
        """
        try:
            if psi is None:
                raise ValueError("psi cannot be None")
            if I_t is None:
                raise ValueError("I_t cannot be None")
            if desires is None:
                raise ValueError("desires cannot be None")
                
            return echozero_dynamics(
                psi=psi,
                t=t,
                I_t=I_t,
                desires=desires,
                node_freqs=self.node_freqs,
                K=self.K,
                alpha=self.alpha,
                beta=self.beta,
                lambda_hub=self.lambda_hub,
            )
        except ValueError as e:
            logger.error(f"EchoZeroSystem forward validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"EchoZeroSystem forward failed: {e}")
            raise


def echozero_dynamics(
    psi: torch.Tensor,
    t: Optional[float],
    I_t: torch.Tensor,
    desires: torch.Tensor,
    node_freqs: torch.Tensor,
    K: torch.Tensor,
    alpha: float = 0.10,
    beta: float = 0.05,
    lambda_hub: float = 0.02,
) -> torch.Tensor:
    """
    Core EchoZero equations of motion (EXACT from spec) with error handling.

    dψ/dt = (-α + iω)ψ + Σ(K@ψ) - β|ψ|²ψ + γψ + I(t) - λΣψ

    Args:
        psi: Resonant state ψ ∈ ℂ^N
        t: Time (optional)
        I_t: Drive signal I(t) ∈ ℂ^N
        desires: Desire vectors ∈ ℝ^N
        node_freqs: Natural frequencies ω ∈ ℝ^N
        K: Coupling matrix K ∈ ℂ^(N×N)
        alpha: Damping coefficient
        beta: Nonlinear damping
        lambda_hub: Hub coupling strength

    Returns:
        dpsi_dt: Time derivative ∈ ℂ^N
    """
    try:
        omega = node_freqs
        local = (-alpha + 1j * omega) * psi
        coupled = torch.matmul(K, psi)
        psi_magnitude_sq = (psi.real ** 2 + psi.imag ** 2)
        nonlinear = -beta * psi_magnitude_sq * psi
        gamma = compute_want_modulation(psi, desires)
        want = gamma * psi
        drive = I_t
        hub = -lambda_hub * psi.sum()
        dpsi_dt = local + coupled + nonlinear + want + drive + hub

        return dpsi_dt
    except Exception as e:
        logger.error(f"echozero_dynamics failed: {e}")
        raise


def compute_want_modulation(
    psi: torch.Tensor,
    desires: torch.Tensor,
    gamma_base: float = 0.2,
    gamma_scale: float = 0.4,
) -> torch.Tensor:
    """
    Compute want modulation coefficient γ from desire alignment with error handling.

    γ = 0.2 + 0.4 * sigmoid(align)

    where align = cosine_similarity(Re(ψ), desires)

    Args:
        psi: Resonant state ψ ∈ ℂ^N
        desires: Desire vectors ∈ ℝ^N
        gamma_base: Base modulation (default 0.2)
        gamma_scale: Scale factor (default 0.4)

    Returns:
        gamma: Want modulation ∈ ℝ^N
    """
    try:
        psi_real = psi.real
        psi_norm = torch.norm(psi_real, dim=-1, keepdim=True) + 1e-8
        desires_norm = torch.norm(desires, dim=-1, keepdim=True) + 1e-8

        align = torch.sum(psi_real * desires, dim=-1) / (psi_norm.squeeze() * desires_norm.squeeze())
        gamma = gamma_base + gamma_scale * torch.sigmoid(align)

        return gamma
    except Exception as e:
        logger.error(f"compute_want_modulation failed: {e}")
        raise


def compute_coherence(
    psi: torch.Tensor,
    node_freqs: torch.Tensor,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """
    Compute attention coherence from resonance with error handling.

    coherence = sigmoid(10 * (1 - |ψ - ω|))

    Args:
        psi: Resonant state ψ ∈ ℂ^N
        node_freqs: Natural frequencies ω ∈ ℝ^N
        sharpness: Sigmoid sharpness (default 10.0)

    Returns:
        coherence: Attention coherence ∈ [0, 1]^N
    """
    try:
        psi_real = psi.real
        distance = torch.abs(psi_real - node_freqs)
        coherence = torch.sigmoid(sharpness * (1.0 - distance))

        return coherence
    except Exception as e:
        logger.error(f"compute_coherence failed: {e}")
        raise


def stability_check(psi: torch.Tensor, max_magnitude: float = 10.0) -> bool:
    """
    Check if state is stable (not diverging) with error handling.

    Args:
        psi: Resonant state ψ ∈ ℂ^N
        max_magnitude: Maximum allowed magnitude

    Returns:
        is_stable: True if stable, False if diverging
    """
    try:
        magnitude = torch.abs(psi)
        max_mag = magnitude.max().item()

        is_stable = (
            torch.isfinite(psi).all().item()
            and max_mag < max_magnitude
        )

        return is_stable
    except Exception as e:
        logger.error(f"stability_check failed: {e}")
        return False


def enforce_hermiticity(K: torch.Tensor) -> torch.Tensor:
    """
    Enforce hermiticity of coupling matrix: K = K† with error handling.

    Args:
        K: Coupling matrix K ∈ ℂ^(N×N)

    Returns:
        K_hermitian: Hermitian matrix
    """
    try:
        K_hermitian = (K + K.conj().T) / 2
        return K_hermitian
    except Exception as e:
        logger.error(f"enforce_hermiticity failed: {e}")
        raise
