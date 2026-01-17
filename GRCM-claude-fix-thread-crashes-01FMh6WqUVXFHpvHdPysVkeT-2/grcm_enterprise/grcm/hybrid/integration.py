"""
Integration layer between GRCM and EchoZero.

Converts GRCM grounded representations into complex-valued drive signals.
"""

import torch
import torch.nn as nn
from typing import Tuple


class FrequencyDrive(nn.Module):
    """
    Converts GRCM grounded representation to complex drive signal I(t).

    Pipeline:
        grounded (15D) → freq (8D) → amplitude η + phase φ → I = η * e^(iφ)
    """

    def __init__(
        self,
        grounded_dim: int = 15,
        freq_dim: int = 8,
        n_nodes: int = 64,
    ):
        """
        Initialize frequency drive converter.

        Args:
            grounded_dim: Dimension of grounded representation
            freq_dim: Intermediate frequency dimension
            n_nodes: Number of EchoZero nodes
        """
        super().__init__()
        self.grounded_dim = grounded_dim
        self.freq_dim = freq_dim
        self.n_nodes = n_nodes

        # Project grounded → frequency space
        self.freq_proj = nn.Linear(grounded_dim, freq_dim)

        # Expand to n_nodes
        self.node_proj = nn.Linear(freq_dim, n_nodes)

    def forward(self, grounded: torch.Tensor) -> torch.Tensor:
        """
        Convert grounded representation to complex drive signal.

        Args:
            grounded: Grounded representation [batch, grounded_dim]

        Returns:
            I_t: Complex drive signal [batch, n_nodes]
        """
        # Project to frequency space with tanh activation
        freq = torch.tanh(self.freq_proj(grounded))  # [batch, freq_dim]

        # Expand to node space
        freq_nodes = self.node_proj(freq)  # [batch, n_nodes]

        # Compute amplitude: η = ||freq||
        eta = torch.norm(freq, dim=-1, keepdim=True)  # [batch, 1]

        # Compute phase: φ = angle(FFT(freq))
        freq_fft = torch.fft.fft(freq, dim=-1)  # Complex FFT
        phi = torch.angle(freq_fft)  # [batch, freq_dim]

        # Expand phase to n_nodes
        if phi.shape[-1] < self.n_nodes:
            # Interpolate or repeat
            phi = torch.nn.functional.interpolate(
                phi.unsqueeze(1),
                size=self.n_nodes,
                mode='linear',
                align_corners=False,
            ).squeeze(1)
        elif phi.shape[-1] > self.n_nodes:
            phi = phi[..., :self.n_nodes]

        # Create complex drive: I = η * e^(iφ)
        I_t = eta * torch.exp(1j * phi)  # [batch, n_nodes]

        # Alternative: use freq_nodes directly
        # I_t = torch.complex(freq_nodes, torch.zeros_like(freq_nodes))

        return I_t


def create_complex_drive(
    grounded: torch.Tensor,
    n_nodes: int,
    method: str = "fft",
) -> torch.Tensor:
    """
    Create complex drive signal from grounded representation.

    Args:
        grounded: Grounded representation
        n_nodes: Number of nodes
        method: Method to use ('fft', 'random_phase', 'real')

    Returns:
        I_t: Complex drive signal
    """
    batch_size = grounded.shape[0]
    device = grounded.device

    if method == "fft":
        # Use FFT for phase
        freq = grounded[:, :min(8, grounded.shape[1])]
        eta = torch.norm(freq, dim=-1, keepdim=True)
        freq_fft = torch.fft.fft(freq, dim=-1)
        phi = torch.angle(freq_fft)

        # Expand to n_nodes
        phi = torch.nn.functional.interpolate(
            phi.unsqueeze(1),
            size=n_nodes,
            mode='linear',
            align_corners=False,
        ).squeeze(1)

        I_t = eta * torch.exp(1j * phi)

    elif method == "random_phase":
        # Random phase
        amplitude = torch.norm(grounded, dim=-1, keepdim=True).expand(-1, n_nodes)
        phase = torch.rand(batch_size, n_nodes, device=device) * 2 * torch.pi
        I_t = amplitude * torch.exp(1j * phase)

    elif method == "real":
        # Pure real drive
        # Project grounded to n_nodes
        if grounded.shape[1] < n_nodes:
            real_part = torch.nn.functional.interpolate(
                grounded.unsqueeze(1),
                size=n_nodes,
                mode='linear',
                align_corners=False,
            ).squeeze(1)
        else:
            real_part = grounded[:, :n_nodes]

        I_t = torch.complex(real_part, torch.zeros_like(real_part))

    else:
        raise ValueError(f"Unknown method: {method}")

    return I_t


def extract_desires_from_grcm(
    desire_output: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """
    Extract desire vectors from GRCM desire module output.

    Args:
        desire_output: GRCM desire vectors [batch, desire_dim]
        n_nodes: Number of EchoZero nodes

    Returns:
        desires: Expanded desires [batch, n_nodes]
    """
    # Expand to match n_nodes
    if desire_output.shape[1] < n_nodes:
        desires = torch.nn.functional.interpolate(
            desire_output.unsqueeze(1),
            size=n_nodes,
            mode='linear',
            align_corners=False,
        ).squeeze(1)
    else:
        desires = desire_output[:, :n_nodes]

    return desires
