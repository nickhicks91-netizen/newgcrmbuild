"""
Spiral Lattice Geometry Layer for EchoZero Memory Systems

Purpose:
  - Adds geometric continuity to memory
  - Encodes distance (time) as radius
  - Encodes recurrence (loops) as angle
  - Enhances multi-turn consistency, narrative flow, and stability

Key Features:
  ✓ Logarithmic inward spiral for long-term memory compression
  ✓ Smooth radial decay → reduces old noise
  ✓ Phase-based rotation → reinforces consistent trajectories
  ✓ Geometric coherence score → for downstream gating
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple


class SpiralLattice(nn.Module):
    """
    Spiral Lattice Geometry Layer for EchoZero Memory Systems.

    Implements a logarithmic spiral manifold for temporal memory encoding:
    - Older memories compress radially toward center
    - Recurrent patterns reinforced through angular momentum
    - Geometric coherence metric for stability gating

    Mathematical Foundation:
    -------------------------
    Spiral coordinates: (r, θ) where:
      r(t) = exp(-λ × t)        [radial decay]
      θ(t) = ω × t              [angular velocity]

    Projection:
      ψ_spiral = r(t) × R(θ) × ψ
      where R(θ) is rotation in feature space

    Coherence:
      C = ⟨ψ, ψ_mem⟩ / (||ψ|| × ||ψ_mem||)

    Complexity:
    -----------
    - Time: O(N) per forward pass
    - Space: O(N × M) where M = memory_length
    - Memory access: Sequential (cache-friendly)
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_length: int = 256,
        radial_decay: float = 0.015,
        angular_velocity: float = 0.45,
        stability_gain: float = 1.6,
    ):
        """
        Initialize Spiral Lattice layer.

        Args:
            hidden_dim: Dimension of hidden state vectors
            memory_length: Length of spiral memory buffer
            radial_decay: Decay rate λ for radial compression (higher = faster decay)
            angular_velocity: Angular rotation rate ω (radians per step)
            stability_gain: Amplification factor for coherence-based stabilization
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.memory_length = memory_length
        self.radial_decay = radial_decay
        self.angular_velocity = angular_velocity
        self.stability_gain = stability_gain

        # Continuous geometric memory buffer
        self.register_buffer(
            "spiral_memory",
            torch.zeros(1, memory_length, hidden_dim)
        )
        self.register_buffer("pointer", torch.tensor(0, dtype=torch.long))

    # ============================================================
    # INTERNAL GEOMETRY HELPERS
    # ============================================================

    def get_radius(self, t: int) -> float:
        """
        Computes radius at time step t.

        Implements exponential decay: r(t) = exp(-λ × t)
        Older memories (larger t) have smaller radius (closer to center).

        Args:
            t: Time step (0 to memory_length-1)

        Returns:
            Radius in range (0, 1]
        """
        return math.exp(-self.radial_decay * t)

    def get_angle(self, t: int) -> float:
        """
        Computes angular position at time step t.

        Implements linear rotation: θ(t) = ω × t
        Creates spiral trajectory through memory space.

        Args:
            t: Time step

        Returns:
            Angle in radians
        """
        return t * self.angular_velocity

    def project_to_spiral(self, vector: torch.Tensor, t: int) -> torch.Tensor:
        """
        Maps a hidden vector into the spiral coordinate system.

        Projects vector onto spiral manifold using:
          1. Rotation by angle θ(t) in feature space
          2. Scaling by radius r(t)

        The rotation is implemented as a 2D rotation generalized to high dimensions:
          rot(ψ) = cos(θ) × ψ + sin(θ) × roll(ψ, 1)

        Args:
            vector: [hidden_dim] tensor to project
            t: Current time step

        Returns:
            Projected vector on spiral manifold
        """
        r = self.get_radius(t)
        θ = self.get_angle(t)

        # Rotation in feature space (generalized 2D rotation)
        cos_t = math.cos(θ)
        sin_t = math.sin(θ)

        # Simple 2-component rotation that generalizes to high-dimensional space
        # This creates a smooth rotational flow through the feature manifold
        rot = (vector * cos_t) + torch.roll(vector, shifts=1, dims=-1) * sin_t

        # Scale by radial distance
        return r * rot

    def compute_geometric_coherence(
        self,
        x: torch.Tensor,
        mem: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the geometric alignment between current state and spiral memory.

        Uses normalized cosine similarity as coherence metric:
          C = ⟨x, m⟩ / (||x|| × ||m||)

        High coherence → current state aligns with spiral trajectory
        Low coherence → current state deviates from expected geometry

        Args:
            x: Current hidden state [Batch, hidden_dim]
            mem: Spiral memory vector [Batch, hidden_dim]

        Returns:
            Coherence score in range [-1, 1] (averaged over batch)
        """
        # Normalized vectors
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        m_norm = mem / (mem.norm(dim=-1, keepdim=True) + 1e-8)

        # Cosine similarity
        coherence = torch.sum(x_norm * m_norm, dim=-1)

        # Average over batch
        return torch.mean(coherence)

    # ============================================================
    # MAIN FORWARD PASS
    # ============================================================

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass through Spiral Lattice layer.

        Process:
          1. Retrieve spiral memory at current pointer
          2. Compute geometric coherence between input and memory
          3. Apply stability boost based on coherence
          4. Project updated state back to spiral manifold
          5. Advance pointer along spiral

        Args:
            x: Input tensor [Batch, hidden_dim]

        Returns:
            output: Geometry-aligned hidden state [Batch, hidden_dim]
            metrics: Dictionary containing:
                - coherence: Alignment with spiral trajectory
                - radius: Current radial position
                - angle: Current angular position (radians)
                - stabilizer: Coherence-based gain factor
        """
        batch = x.size(0)
        t = int(self.pointer.item())

        # Retrieve projected memory at current pointer
        mem_t = self.spiral_memory[:, t, :].expand(batch, -1)

        # Compute geometric coherence
        coherence = self.compute_geometric_coherence(x, mem_t)

        # Stability boost: encourages states aligned with spiral trajectory
        # Higher coherence → stronger amplification
        stabilizer = 1.0 + self.stability_gain * coherence.item()

        # Output after stabilizer application
        output = x * stabilizer

        # Project updated memory into spiral manifold
        # Use first batch element for memory update (assuming batch coherence)
        projected = self.project_to_spiral(output[0].detach(), t)
        self.spiral_memory[:, t, :] = projected

        # Advance pointer along spiral (wraps at memory_length)
        self.pointer = (self.pointer + 1) % self.memory_length

        # Return output and diagnostic metrics
        return output, {
            "coherence": float(coherence.item()),
            "radius": float(self.get_radius(t)),
            "angle": float(self.get_angle(t)),
            "stabilizer": float(stabilizer),
        }


def create_spiral_lattice(
    hidden_dim: int,
    memory_length: int = 256,
    radial_decay: float = 0.015,
    angular_velocity: float = 0.45,
    stability_gain: float = 1.6,
) -> SpiralLattice:
    """
    Factory function for creating SpiralLattice instances.

    Convenience wrapper with sensible defaults for common use cases.

    Args:
        hidden_dim: Dimension of hidden state
        memory_length: Spiral buffer length (default: 256)
        radial_decay: Radial compression rate (default: 0.015)
        angular_velocity: Rotation rate in radians/step (default: 0.45)
        stability_gain: Coherence amplification factor (default: 1.6)

    Returns:
        Configured SpiralLattice module

    Example:
        >>> spiral = create_spiral_lattice(hidden_dim=64)
        >>> x = torch.randn(1, 64)
        >>> output, metrics = spiral(x)
        >>> print(f"Coherence: {metrics['coherence']:.3f}")
    """
    return SpiralLattice(
        hidden_dim=hidden_dim,
        memory_length=memory_length,
        radial_decay=radial_decay,
        angular_velocity=angular_velocity,
        stability_gain=stability_gain,
    )
