"""
Coupling matrix K generation for EchoZero.

Converts adjacency list to complex-valued coupling matrix.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


class CouplingMatrix:
    """
    Generator for coupling matrix K ∈ ℂ^(N×N).

    The coupling matrix defines the network topology and interaction strength
    between oscillators. Must be Hermitian for energy conservation.
    """

    def __init__(
        self,
        n_nodes: int,
        device: str = "cpu",
    ):
        """
        Initialize coupling matrix builder.

        Args:
            n_nodes: Number of nodes
            device: Computation device
        """
        self.n_nodes = n_nodes
        self.device = device

    def from_adjacency(
        self,
        edges: List[Tuple[int, int]],
        coupling_strength: float = 1.0,
        phase_randomize: bool = False,
        hermitian: bool = True,
    ) -> torch.Tensor:
        """
        Build coupling matrix from edge list.

        Args:
            edges: List of (source, target) tuples
            coupling_strength: Base coupling strength
            phase_randomize: Add random phase to couplings
            hermitian: Enforce hermiticity K = K†

        Returns:
            K: Coupling matrix K ∈ ℂ^(N×N)
        """
        # Initialize as complex zeros
        K = torch.zeros(
            (self.n_nodes, self.n_nodes),
            dtype=torch.complex64,
            device=self.device,
        )

        for src, tgt in edges:
            if phase_randomize:
                # Random phase: k_ij = |k| * e^(iφ)
                phase = np.random.uniform(0, 2 * np.pi)
                coupling = coupling_strength * np.exp(1j * phase)
            else:
                # Real coupling
                coupling = coupling_strength

            K[src, tgt] = coupling

        # Enforce hermiticity
        if hermitian:
            K = (K + K.conj().T) / 2

        return K

    def add_random_phase(
        self,
        K: torch.Tensor,
        phase_std: float = 0.1,
    ) -> torch.Tensor:
        """
        Add random phase modulation to coupling matrix.

        Args:
            K: Existing coupling matrix
            phase_std: Standard deviation of phase perturbation

        Returns:
            K_perturbed: Perturbed coupling matrix
        """
        # Generate random phases
        phases = torch.randn_like(K.real) * phase_std

        # Apply phase rotation
        phase_factor = torch.exp(1j * phases)
        K_perturbed = K * phase_factor

        # Re-enforce hermiticity
        K_perturbed = (K_perturbed + K_perturbed.conj().T) / 2

        return K_perturbed

    def normalize(
        self,
        K: torch.Tensor,
        target_norm: float = 1.0,
    ) -> torch.Tensor:
        """
        Normalize coupling matrix by Frobenius norm.

        Args:
            K: Coupling matrix
            target_norm: Target Frobenius norm

        Returns:
            K_normalized: Normalized matrix
        """
        current_norm = torch.norm(K, p="fro")
        K_normalized = K * (target_norm / (current_norm + 1e-8))
        return K_normalized


def build_coupling_matrix(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    coupling_strength: float = 1.0,
    phase_randomize: bool = False,
    normalize: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convenience function to build coupling matrix.

    Args:
        n_nodes: Number of nodes
        edges: List of edge tuples
        coupling_strength: Base coupling strength
        phase_randomize: Add random phases
        normalize: Normalize by Frobenius norm
        device: Computation device

    Returns:
        K: Coupling matrix K ∈ ℂ^(N×N)
    """
    builder = CouplingMatrix(n_nodes, device)

    K = builder.from_adjacency(
        edges=edges,
        coupling_strength=coupling_strength,
        phase_randomize=phase_randomize,
        hermitian=True,
    )

    if normalize:
        K = builder.normalize(K, target_norm=float(n_nodes) ** 0.5)

    return K


def verify_hermiticity(K: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Verify that K = K†.

    Args:
        K: Coupling matrix
        tol: Tolerance for equality

    Returns:
        is_hermitian: True if hermitian
    """
    K_conj_T = K.conj().T
    diff = torch.abs(K - K_conj_T).max().item()
    return diff < tol


def coupling_spectrum(K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute eigenvalues and eigenvectors of coupling matrix.

    For hermitian K, eigenvalues are real.

    Args:
        K: Coupling matrix

    Returns:
        eigenvalues: Real eigenvalues
        eigenvectors: Complex eigenvectors
    """
    # Convert to numpy for eigen decomposition
    K_np = K.cpu().numpy()

    eigenvalues, eigenvectors = np.linalg.eigh(K_np)

    eigenvalues = torch.from_numpy(eigenvalues).to(K.device)
    eigenvectors = torch.from_numpy(eigenvectors).to(K.device)

    return eigenvalues, eigenvectors


def analyze_coupling(K: torch.Tensor) -> dict:
    """
    Analyze properties of coupling matrix.

    Args:
        K: Coupling matrix

    Returns:
        stats: Dictionary of statistics
    """
    is_hermitian = verify_hermiticity(K)
    eigenvalues, _ = coupling_spectrum(K)

    stats = {
        "is_hermitian": is_hermitian,
        "frobenius_norm": torch.norm(K, p="fro").item(),
        "spectral_radius": eigenvalues.abs().max().item(),
        "min_eigenvalue": eigenvalues.min().item(),
        "max_eigenvalue": eigenvalues.max().item(),
        "spectral_gap": (eigenvalues.max() - eigenvalues.min()).item(),
    }

    return stats
