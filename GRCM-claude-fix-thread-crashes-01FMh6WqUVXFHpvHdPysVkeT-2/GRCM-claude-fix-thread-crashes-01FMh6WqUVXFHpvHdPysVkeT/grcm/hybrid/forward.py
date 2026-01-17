"""
Unified forward pass combining EchoZero + GRCM.

This is the core integration module that implements the exact specification.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np

from ..modules.grounding import GroundingLayer
from ..modules.qualia import QualiaModule
from ..modules.phi import PhiEstimator
from ..modules.memory import MemoryGrid
from ..modules.desire import DesireModule
from ..echozero import (
    EchoZeroSystem,
    compute_coherence,
    compute_want_modulation,
)
from ..echozero.ode_solver import integrate_echozero
from ..echozero.mobius import MobiusEchoLayer
from ..echozero.identity import TorsionLattice3D, LatticeUpdater
from .integration import FrequencyDrive, extract_desires_from_grcm


class EchoGRCMHybrid(nn.Module):
    """
    Unified EchoZero + GRCM hybrid system.

    Implements the exact forward pass from the canonical specification:
        1. GRCM grounding
        2. Frequency projection & complex drive
        3. EchoZero integration
        4. Coherence & qualia
        5. Phi calculation
        6. Memory update
        7. Proprioceptive force
    """

    def __init__(
        self,
        n_nodes: int = 64,
        grounded_dim: int = 15,
        memory_dim: int = 128,
        prop_dim: int = 6,
        coupling_matrix: Optional[torch.Tensor] = None,
        node_freqs: Optional[torch.Tensor] = None,
        dt: float = 0.01,
        integration_steps: int = 10,
        device: str = "cpu",
        # Torsion Lattice parameters (SLOW LOOP - optional)
        enable_torsion_lattice: bool = True,
        lattice_size: int = 5,
        lattice_coupling: float = 0.8,
        lattice_decay: float = 0.03,
        lattice_write_strength: float = 0.03,
        lattice_sync_interval: int = 300,
        lattice_torsion_threshold: float = 0.2,
    ):
        """
        Initialize EchoGRCM hybrid system.

        Args:
            n_nodes: Number of EchoZero oscillator nodes
            grounded_dim: Dimension of grounded representation
            memory_dim: Memory state dimension
            prop_dim: Proprioceptive state dimension
            coupling_matrix: EchoZero coupling matrix (if None, created)
            node_freqs: Natural frequencies (if None, created)
            dt: Integration time step
            integration_steps: Number of integration steps
            device: Computation device
            enable_torsion_lattice: Enable 3D torsion memory lattice (slow loop)
            lattice_size: Torsion lattice dimension (size³ grid)
            lattice_coupling: XY coupling strength
            lattice_decay: Self-healing decay rate
            lattice_write_strength: Weak write coefficient (< 0.05)
            lattice_sync_interval: Steps between syncs (controls Hz)
            lattice_torsion_threshold: Möbius gate threshold
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.grounded_dim = grounded_dim
        self.memory_dim = memory_dim
        self.prop_dim = prop_dim
        self.dt = dt
        self.integration_steps = integration_steps
        self.device = device

        # GRCM modules
        self.grounding = GroundingLayer(input_dim=grounded_dim)
        self.desire = DesireModule(input_dim=grounded_dim, desire_dim=n_nodes)
        self.memory = MemoryGrid(memory_size=memory_dim, input_dim=n_nodes)
        self.qualia = QualiaModule(input_dim=n_nodes)
        self.phi = PhiEstimator()

        # EchoZero integration
        self.freq_drive = FrequencyDrive(
            grounded_dim=grounded_dim,
            freq_dim=8,
            n_nodes=n_nodes,
        )

        # Initialize EchoZero system
        if coupling_matrix is None:
            from ..echozero import build_ring_lattice, build_coupling_matrix

            adjacency, node_freqs_gen, edges = build_ring_lattice(
                n_nodes=n_nodes,
                add_torsion=True,
                device=device,
            )
            coupling_matrix = build_coupling_matrix(
                n_nodes=n_nodes,
                edges=edges,
                device=device,
            )
            node_freqs = node_freqs_gen

        self.echozero = EchoZeroSystem(
            n_nodes=n_nodes,
            coupling_matrix=coupling_matrix,
            node_freqs=node_freqs,
            device=device,
        )

        # Linear layer for qualia projection
        self.qualia_proj = nn.Linear(n_nodes, 4)

        # Proprioceptive force calculation
        self.force_scale = 0.1

        # Möbius layer for torsion score computation
        self.mobius = MobiusEchoLayer(
            hidden_dim=n_nodes,
            loop_len=256,
            device=device,
        )

        # 3D Torsion Memory Lattice (SLOW LOOP - optional)
        self.enable_torsion_lattice = enable_torsion_lattice
        if enable_torsion_lattice:
            self.torsion_lattice = TorsionLattice3D(
                size=lattice_size,
                coupling=lattice_coupling,
                gamma=lattice_decay,
            )
            self.lattice_updater = LatticeUpdater(
                lattice=self.torsion_lattice,
                write_strength=lattice_write_strength,
                sync_interval=lattice_sync_interval,
                torsion_threshold=lattice_torsion_threshold,
            )
        else:
            self.torsion_lattice = None
            self.lattice_updater = None

        # Move to device
        self.to(device)

    def forward(
        self,
        image_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        action: torch.Tensor,
        prop_state: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        psi: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass (EXACT from specification).

        Args:
            image_emb: Visual embeddings [batch, 512]
            audio_emb: Audio embeddings [batch, 768]
            action: Action vector [batch, 4]
            prop_state: Proprioceptive state [batch, 6] (optional)
            memory: Memory state [batch, 128] (optional)
            psi: Initial EchoZero state [batch, n_nodes] (optional)

        Returns:
            Dictionary with:
                - psi: Final resonant state
                - coherence: Attention coherence
                - qualia: Conscious states
                - phi: Integrated information
                - memory: Updated memory
                - prop_state: Updated proprioception
                - desires: Desire vectors
                - gamma: Want modulation
        """
        batch_size = image_emb.shape[0]

        # Initialize states if not provided
        if prop_state is None:
            prop_state = torch.zeros(batch_size, self.prop_dim, device=self.device)

        if memory is None:
            memory = torch.zeros(batch_size, self.memory_dim, device=self.device)

        if psi is None:
            # Random initial state
            psi_real = torch.randn(batch_size, self.n_nodes, device=self.device) * 0.1
            psi_imag = torch.randn(batch_size, self.n_nodes, device=self.device) * 0.1
            psi = torch.complex(psi_real, psi_imag)

        # 1. GRCM Grounding Layer
        grounded = self.grounding(image_emb, audio_emb, prop_state)  # [batch, 15]

        # 2. Frequency Projection & Complex Drive
        I_t = self.freq_drive(grounded)  # [batch, n_nodes] complex

        # 3. Desire Generation
        desires = self.desire(grounded)  # [batch, n_nodes]

        # 4. EchoZero Integration
        # Integrate for multiple steps
        t_span = (0.0, self.dt * self.integration_steps)

        # For batched operation, process each batch element
        psi_final_list = []
        for i in range(batch_size):
            psi_final, _, _ = integrate_echozero(
                psi0=psi[i],
                t_span=t_span,
                I_t=I_t[i],
                desires=desires[i],
                node_freqs=self.echozero.node_freqs,
                K=self.echozero.K,
                dt=self.dt,
                device=self.device,
            )
            psi_final_list.append(psi_final)

        psi = torch.stack(psi_final_list)  # [batch, n_nodes]

        # 5. Coherence Calculation
        coherence = compute_coherence(psi, self.echozero.node_freqs)  # [batch, n_nodes]

        # 6. Qualia Generation
        psi_real = psi.real
        qualia = torch.softmax(self.qualia_proj(psi_real), dim=-1)  # [batch, 4]

        # 7. Integrated Information (Φ)
        phi = self._compute_phi(psi, coherence, memory, qualia)  # [batch]

        # 8. Want Modulation
        gamma = compute_want_modulation(psi, desires)  # [batch, n_nodes]

        # 9. Memory Update (GRU)
        memory_input = psi_real * coherence  # Weight by coherence
        memory = self.memory(memory_input, memory)  # [batch, 128]

        # 10. Proprioceptive Force
        # force = align.mean() * qualia[:, 1]  # Alert qualia
        align = torch.cosine_similarity(
            psi_real,
            desires,
            dim=-1,
        ).mean(dim=-1, keepdim=True)  # [batch, 1]

        force = align * qualia[:, 1:2] * self.force_scale  # [batch, 1]

        # Update proprioceptive state (simple integration)
        # Assuming prop_state = [position(3), velocity(3)]
        # Update velocity with force
        prop_state = prop_state + torch.cat([
            torch.zeros(batch_size, 3, device=self.device),
            force.expand(-1, 3),
        ], dim=-1) * self.dt

        # 11. Möbius Torsion Score (for stability gating)
        psi_validated, mobius_metrics = self.mobius(psi_real)
        torsion_score = mobius_metrics['energy'].mean().item()

        # 12. Torsion Lattice Sync (SLOW LOOP - non-blocking)
        lattice_sync_info = None
        if self.enable_torsion_lattice:
            # Extract identity state as 2D vector (mean of real + imag components)
            identity_state = np.array([
                psi_real.mean().item(),
                psi.imag.mean().item(),
            ])

            # Periodic sync (only commits every N steps if torsion is low)
            lattice_sync_info = self.lattice_updater.maybe_sync(
                echo_state=identity_state,
                torsion_score=torsion_score,
            )

        return {
            "psi": psi,
            "coherence": coherence,
            "qualia": qualia,
            "phi": phi,
            "memory": memory,
            "prop_state": prop_state,
            "desires": desires,
            "gamma": gamma,
            "grounded": grounded,
            "mobius_metrics": mobius_metrics,
            "torsion_score": torsion_score,
            "lattice_sync_info": lattice_sync_info,
        }

    def _compute_phi(
        self,
        psi: torch.Tensor,
        coherence: torch.Tensor,
        memory: torch.Tensor,
        qualia: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute integrated information Φ (exact from spec).

        Φ = Var(ψ) × mean(coherence) + log(1 + ||memory||) + max(qualia)

        Args:
            psi: Resonant state [batch, n_nodes]
            coherence: Coherence [batch, n_nodes]
            memory: Memory state [batch, memory_dim]
            qualia: Qualia states [batch, 4]

        Returns:
            phi: Integrated information [batch]
        """
        # Variance of ψ (using real part)
        psi_var = torch.var(psi.real, dim=-1)  # [batch]

        # Mean coherence
        coherence_mean = coherence.mean(dim=-1)  # [batch]

        # Memory integration
        memory_norm = torch.norm(memory, dim=-1)  # [batch]
        memory_term = torch.log(1.0 + memory_norm)  # [batch]

        # Qualia richness
        qualia_max = qualia.max(dim=-1)[0]  # [batch]

        # Combined Φ
        phi = (
            psi_var * coherence_mean
            + memory_term
            + qualia_max
        )

        return phi

    def reset_state(self):
        """Reset EchoZero and memory states."""
        self.echozero.reset()

    def get_system_info(self) -> Dict[str, any]:
        """Get system configuration and state info."""
        return {
            "n_nodes": self.n_nodes,
            "grounded_dim": self.grounded_dim,
            "memory_dim": self.memory_dim,
            "dt": self.dt,
            "integration_steps": self.integration_steps,
            "coupling_spectrum": self.echozero.K.abs().mean().item(),
        }


def unified_forward(
    model: EchoGRCMHybrid,
    image_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    action: torch.Tensor,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function for unified forward pass.

    Args:
        model: EchoGRCM hybrid model
        image_emb: Visual embeddings
        audio_emb: Audio embeddings
        action: Action vector
        **kwargs: Additional arguments

    Returns:
        Output dictionary
    """
    return model(image_emb, audio_emb, action, **kwargs)
