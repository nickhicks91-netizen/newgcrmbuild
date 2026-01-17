"""
Body Simulator - Proprioceptive Embodiment
Simple physics simulation for grounded agency
"""
import torch
import torch.nn.functional as F
from typing import Optional
from ..config import BodyConfig


class BodySimulator:
    """
    Simplified physics-based body simulator.

    State vector: [position (8d), velocity (8d)] = 16d total

    Dynamics:
    - Force = desire_alignment * action
    - Acceleration = Force / mass
    - Velocity += Acceleration * dt
    - Position += Velocity * dt

    Provides proprioceptive feedback for embodied cognition.
    """

    def __init__(self, config: Optional[BodyConfig] = None):
        self.config = config or BodyConfig()

        # State: [position, velocity]
        self.state = torch.zeros(1, self.config.state_dim)

        # Physics parameters
        self.dt = self.config.dt
        self.mass = self.config.mass

        # Split dimensions
        self.pos_dim = self.config.state_dim // 2
        self.vel_dim = self.config.state_dim - self.pos_dim

    def update(
        self,
        action: torch.Tensor,
        desire_align: torch.Tensor
    ) -> torch.Tensor:
        """
        Update body state based on action and desire

        Args:
            action: Action vector [batch, action_dim]
            desire_align: Desire alignment [batch, 1]

        Returns:
            New state [batch, state_dim]
        """
        # Pad action if needed
        if action.size(1) < self.pos_dim:
            action = F.pad(action, (0, self.pos_dim - action.size(1)))

        # Compute force (desire modulates action)
        force = desire_align * action[:, :self.pos_dim]  # [batch, pos_dim]

        # Compute acceleration
        accel = force / self.mass

        # Clone state for gradient safety
        self.state = self.state.clone()

        # Update velocity (second half of state)
        self.state[:, self.pos_dim:] += accel * self.dt

        # Update position (first half of state)
        self.state[:, :self.pos_dim] += self.state[:, self.pos_dim:] * self.dt

        return self.state.clone()

    def get_state(self) -> torch.Tensor:
        """Get current body state"""
        return self.state.clone()

    def get_position(self) -> torch.Tensor:
        """Get position component"""
        return self.state[:, :self.pos_dim].clone()

    def get_velocity(self) -> torch.Tensor:
        """Get velocity component"""
        return self.state[:, self.pos_dim:].clone()

    def reset(self, initial_state: Optional[torch.Tensor] = None) -> None:
        """
        Reset body state

        Args:
            initial_state: Optional initial state [batch, state_dim]
        """
        if initial_state is not None:
            self.state = initial_state.clone()
        else:
            self.state = torch.zeros(1, self.config.state_dim)

    def get_body_stats(self) -> dict:
        """
        Get body statistics

        Returns:
            Dictionary with body metrics
        """
        pos = self.get_position()
        vel = self.get_velocity()

        return {
            'position_norm': pos.norm().item(),
            'velocity_norm': vel.norm().item(),
            'position_mean': pos.mean().item(),
            'velocity_mean': vel.mean().item(),
            'kinetic_energy': (0.5 * self.mass * vel.pow(2).sum()).item()
        }

    def apply_damping(self, damping_factor: float = 0.99) -> None:
        """
        Apply velocity damping (friction)

        Args:
            damping_factor: Velocity multiplier (< 1.0)
        """
        self.state[:, self.pos_dim:] *= damping_factor
