"""
Photonic hardware mapping for EchoZero.

Maps resonant dynamics to silicon nitride (SiN) microring resonators (MRRs).

Physical implementation:
- Each oscillator ψ_i → MRR with resonance frequency ω_i
- Coupling K_ij → Directional couplers between waveguides
- Damping α → Waveguide loss + outcoupling
- Nonlinearity β → Kerr effect (n2)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PhotonicDevice:
    """Physical parameters for photonic implementation."""

    # Ring parameters
    radius: float  # Ring radius (μm)
    width: float   # Waveguide width (nm)
    thickness: float  # Waveguide thickness (nm)

    # Material (SiN)
    n_eff: float  # Effective index
    n2: float     # Nonlinear index (m^2/W)
    loss_db_cm: float  # Propagation loss (dB/cm)

    # Coupling
    coupling_gap: float  # Gap to bus waveguide (nm)
    coupling_coeff: float  # Field coupling coefficient

    # Derived
    fsr: float  # Free spectral range (GHz)
    q_factor: float  # Quality factor
    finesse: float  # Finesse


class PhotonicMapper:
    """
    Map EchoZero dynamics to photonic circuits.

    Design flow:
    1. Map node frequencies → MRR resonances
    2. Map coupling matrix → Directional coupler array
    3. Map α, β, λ → Physical loss and nonlinearity
    4. Generate layout coordinates
    """

    def __init__(
        self,
        wavelength: float = 1550e-9,  # Telecom C-band (m)
        device: str = "cpu",
    ):
        """
        Initialize photonic mapper.

        Args:
            wavelength: Operating wavelength (m)
            device: Computation device
        """
        self.wavelength = wavelength
        self.device = device

        # SiN material parameters (typical)
        self.n_eff = 1.8  # Effective index
        self.n2 = 2.4e-19  # Kerr coefficient (m^2/W)
        self.loss_db_cm = 0.1  # Low-loss SiN

    def map_frequency_to_radius(
        self,
        freq: float,
        mode: int = 1,
    ) -> float:
        """
        Map resonant frequency to ring radius.

        For ring resonator:
            ω_resonance = m * c / (n_eff * 2πR)
        where m is mode number.

        Args:
            freq: Normalized frequency (dimensionless)
            mode: Mode number

        Returns:
            radius: Ring radius (μm)
        """
        c = 3e8  # Speed of light (m/s)

        # Map normalized freq to wavelength shift
        # freq ~ 1.0 → wavelength
        wavelength = self.wavelength * (1.0 / freq)

        # Radius for mode m
        radius = (mode * wavelength) / (2 * np.pi * self.n_eff)

        # Convert to μm
        radius_um = radius * 1e6

        # Typical range: 5-50 μm
        radius_um = np.clip(radius_um, 5.0, 50.0)

        return radius_um

    def map_coupling_to_gap(
        self,
        coupling_strength: float,
    ) -> float:
        """
        Map coupling strength to waveguide gap.

        Evanescent coupling: κ ∝ exp(-gap/λ_decay)

        Args:
            coupling_strength: Normalized coupling

        Returns:
            gap: Waveguide gap (nm)
        """
        # Typical decay length ~ 100 nm
        lambda_decay = 100.0  # nm

        # Map: |κ| = 1 → gap = 200 nm (strong coupling)
        #      |κ| = 0.1 → gap = 400 nm (weak coupling)
        gap = 200.0 - 100.0 * np.log(np.abs(coupling_strength) + 0.1)

        # Clip to fabrication limits
        gap = np.clip(gap, 150.0, 500.0)

        return gap

    def generate_layout(
        self,
        node_freqs: torch.Tensor,
        K: torch.Tensor,
    ) -> Dict[str, any]:
        """
        Generate photonic chip layout.

        Args:
            node_freqs: Node frequencies
            K: Coupling matrix

        Returns:
            layout: Dictionary with device specifications
        """
        n_nodes = node_freqs.shape[0]

        devices = []

        for i in range(n_nodes):
            freq = node_freqs[i].item()
            radius = self.map_frequency_to_radius(freq)

            device = PhotonicDevice(
                radius=radius,
                width=500.0,  # nm
                thickness=220.0,  # nm
                n_eff=self.n_eff,
                n2=self.n2,
                loss_db_cm=self.loss_db_cm,
                coupling_gap=250.0,  # nm (default)
                coupling_coeff=0.1,
                fsr=c / (2 * np.pi * radius * 1e-6 * self.n_eff) / 1e9,  # GHz
                q_factor=1e6,  # High-Q SiN
                finesse=100.0,
            )

            devices.append(device)

        # Coupling specs
        couplings = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                k_ij = K[i, j].abs().item()
                if k_ij > 1e-3:  # Threshold for physical coupling
                    gap = self.map_coupling_to_gap(k_ij)
                    couplings.append({
                        "ring_i": i,
                        "ring_j": j,
                        "gap_nm": gap,
                        "strength": k_ij,
                    })

        layout = {
            "n_rings": n_nodes,
            "devices": devices,
            "couplings": couplings,
            "wavelength_nm": self.wavelength * 1e9,
            "material": "Si3N4",
            "substrate": "SiO2",
        }

        return layout

    def estimate_power(
        self,
        n_nodes: int,
        power_per_ring: float = 1e-3,  # 1 mW per ring
    ) -> Dict[str, float]:
        """
        Estimate power consumption.

        Args:
            n_nodes: Number of rings
            power_per_ring: Power per ring (W)

        Returns:
            power_stats: Power estimates
        """
        total_power = n_nodes * power_per_ring

        return {
            "total_power_mw": total_power * 1e3,
            "power_per_ring_mw": power_per_ring * 1e3,
            "n_rings": n_nodes,
        }


def map_to_photonic(
    node_freqs: torch.Tensor,
    K: torch.Tensor,
    alpha: float,
    beta: float,
) -> Dict[str, any]:
    """
    Convenience function to map EchoZero to photonic hardware.

    Args:
        node_freqs: Natural frequencies
        K: Coupling matrix
        alpha: Damping
        beta: Nonlinearity

    Returns:
        hardware_spec: Complete hardware specification
    """
    mapper = PhotonicMapper()

    layout = mapper.generate_layout(node_freqs, K)
    power = mapper.estimate_power(node_freqs.shape[0])

    # Map α → loss
    loss_coefficient = alpha  # Direct mapping

    # Map β → Kerr nonlinearity
    required_n2 = beta * 1e-19  # Scaling

    hardware_spec = {
        "layout": layout,
        "power": power,
        "loss_alpha": loss_coefficient,
        "nonlinearity_beta": beta,
        "required_n2": required_n2,
        "fabrication": "CMOS-compatible SiN photonics",
        "foundry": "LIGENTEC, AMF, or similar",
    }

    return hardware_spec
