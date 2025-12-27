"""
EchoZero Dynamics - Core equations of motion.

Implements the exact resonant dynamics from the canonical specification:
    dψ/dt = local + coupled + nonlinear + want + drive + hub

NO backpropagation. Pure forward dynamics.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
import numpy as np


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
        Initialize EchoZero system.

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
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.beta = beta
        self.lambda_hub = lambda_hub
        self.device = device

        # Register buffers (non-trainable for EchoMirror)
        self.register_buffer("K", coupling_matrix.to(device))
        self.register_buffer("node_freqs", node_freqs.to(device))

        # Initialize state
        self.psi = None
        self.reset()

    def reset(self):
        """Reset to random initial state."""
        # Initialize with small random complex values
        real = torch.randn(self.n_nodes, device=self.device) * 0.1
        imag = torch.randn(self.n_nodes, device=self.device) * 0.1
        self.psi = torch.complex(real, imag)

    def forward(
        self,
        psi: torch.Tensor,
        I_t: torch.Tensor,
        desires: torch.Tensor,
        t: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute dψ/dt according to EchoZero dynamics.

        Args:
            psi: Current state ψ ∈ ℂ^N
            I_t: Drive signal I(t) ∈ ℂ^N
            desires: Desire vectors ∈ ℝ^N
            t: Time (optional, for time-dependent drive)

        Returns:
            dpsi_dt: Time derivative dψ/dt ∈ ℂ^N
        """
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
    Core EchoZero equations of motion (EXACT from spec).

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
    # 1. Local Dynamics: (-α + iω)ψ
    omega = node_freqs
    local = (-alpha + 1j * omega) * psi

    # 2. Coupling: Σ_j K_ij ψ_j
    coupled = torch.matmul(K, psi)

    # 3. Nonlinearity: -β|ψ|²ψ
    psi_magnitude_sq = (psi.real ** 2 + psi.imag ** 2)
    nonlinear = -beta * psi_magnitude_sq * psi

    # 4. Want Modulation: γψ
    gamma = compute_want_modulation(psi, desires)
    want = gamma * psi

    # 5. Drive: I(t)
    drive = I_t

    # 6. Hub Constraint: -λΣψ
    hub = -lambda_hub * psi.sum()

    # Total derivative
    dpsi_dt = local + coupled + nonlinear + want + drive + hub

    return dpsi_dt


def compute_want_modulation(
    psi: torch.Tensor,
    desires: torch.Tensor,
    gamma_base: float = 0.2,
    gamma_scale: float = 0.4,
) -> torch.Tensor:
    """
    Compute want modulation coefficient γ from desire alignment.

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
    # Extract real part
    psi_real = psi.real

    # Compute cosine similarity
    psi_norm = torch.norm(psi_real, dim=-1, keepdim=True) + 1e-8
    desires_norm = torch.norm(desires, dim=-1, keepdim=True) + 1e-8

    align = torch.sum(psi_real * desires, dim=-1) / (psi_norm.squeeze() * desires_norm.squeeze())

    # γ = 0.2 + 0.4 * sigmoid(align)
    gamma = gamma_base + gamma_scale * torch.sigmoid(align)

    return gamma


def compute_coherence(
    psi: torch.Tensor,
    node_freqs: torch.Tensor,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """
    Compute attention coherence from resonance.

    coherence = sigmoid(10 * (1 - |ψ - ω|))

    Args:
        psi: Resonant state ψ ∈ ℂ^N
        node_freqs: Natural frequencies ω ∈ ℝ^N
        sharpness: Sigmoid sharpness (default 10.0)

    Returns:
        coherence: Attention coherence ∈ [0, 1]^N
    """
    # Compute |ψ - ω| (treating ω as real for distance)
    psi_real = psi.real
    distance = torch.abs(psi_real - node_freqs)

    # coherence = sigmoid(10 * (1 - distance))
    coherence = torch.sigmoid(sharpness * (1.0 - distance))

    return coherence


def stability_check(psi: torch.Tensor, max_magnitude: float = 10.0) -> bool:
    """
    Check if state is stable (not diverging).

    Args:
        psi: Resonant state ψ ∈ ℂ^N
        max_magnitude: Maximum allowed magnitude

    Returns:
        is_stable: True if stable, False if diverging
    """
    magnitude = torch.abs(psi)
    max_mag = magnitude.max().item()

    is_stable = (
        torch.isfinite(psi).all().item()
        and max_mag < max_magnitude
    )

    return is_stable


def enforce_hermiticity(K: torch.Tensor) -> torch.Tensor:
    """
    Enforce hermiticity of coupling matrix: K = K†.

    Args:
        K: Coupling matrix K ∈ ℂ^(N×N)

    Returns:
        K_hermitian: Hermitian matrix
    """
    K_hermitian = (K + K.conj().T) / 2
    return K_hermitian
