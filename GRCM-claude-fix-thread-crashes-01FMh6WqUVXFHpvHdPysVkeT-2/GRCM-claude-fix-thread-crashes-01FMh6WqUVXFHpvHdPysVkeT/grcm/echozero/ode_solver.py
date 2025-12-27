"""
ODE solver for EchoZero dynamics.

Implements RK4 and optional torchdiffeq integration for complex-valued ODEs.
"""

import torch
from typing import Callable, Optional, Tuple
import numpy as np


class RK4Solver:
    """
    Fourth-order Runge-Kutta solver for complex-valued ODEs.

    Integrates dψ/dt = f(ψ, t) using classical RK4 scheme:
        k1 = f(ψ_n, t_n)
        k2 = f(ψ_n + k1*dt/2, t_n + dt/2)
        k3 = f(ψ_n + k2*dt/2, t_n + dt/2)
        k4 = f(ψ_n + k3*dt, t_n + dt)
        ψ_{n+1} = ψ_n + (k1 + 2*k2 + 2*k3 + k4) * dt/6
    """

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
    ):
        """
        Initialize RK4 solver.

        Args:
            dt: Time step size
            device: Computation device
        """
        self.dt = dt
        self.device = device

    def step(
        self,
        psi: torch.Tensor,
        t: float,
        f: Callable,
        **kwargs,
    ) -> Tuple[torch.Tensor, float]:
        """
        Single RK4 integration step.

        Args:
            psi: Current state ψ_n
            t: Current time t_n
            f: Derivative function f(ψ, t, **kwargs)
            **kwargs: Additional arguments for f

        Returns:
            psi_next: Next state ψ_{n+1}
            t_next: Next time t_{n+1}
        """
        dt = self.dt

        # k1 = f(ψ_n, t_n)
        k1 = f(psi, t, **kwargs)

        # k2 = f(ψ_n + k1*dt/2, t_n + dt/2)
        k2 = f(psi + k1 * dt / 2, t + dt / 2, **kwargs)

        # k3 = f(ψ_n + k2*dt/2, t_n + dt/2)
        k3 = f(psi + k2 * dt / 2, t + dt / 2, **kwargs)

        # k4 = f(ψ_n + k3*dt, t_n + dt)
        k4 = f(psi + k3 * dt, t + dt, **kwargs)

        # ψ_{n+1} = ψ_n + (k1 + 2*k2 + 2*k3 + k4) * dt/6
        psi_next = psi + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        t_next = t + dt

        return psi_next, t_next

    def integrate(
        self,
        psi0: torch.Tensor,
        t_span: Tuple[float, float],
        f: Callable,
        n_steps: Optional[int] = None,
        save_trajectory: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Integrate over time interval.

        Args:
            psi0: Initial state ψ(t0)
            t_span: (t_start, t_end)
            f: Derivative function
            n_steps: Number of steps (if None, computed from dt)
            save_trajectory: Whether to save full trajectory
            **kwargs: Additional arguments for f

        Returns:
            psi_final: Final state ψ(t_end)
            trajectory: Full trajectory if save_trajectory=True, else None
            times: Time points if save_trajectory=True, else None
        """
        t_start, t_end = t_span

        if n_steps is None:
            n_steps = int((t_end - t_start) / self.dt)

        # Recompute dt to match exactly
        dt_actual = (t_end - t_start) / n_steps
        original_dt = self.dt
        self.dt = dt_actual

        psi = psi0.clone()
        t = t_start

        if save_trajectory:
            trajectory = [psi.clone()]
            times = [t]

        for _ in range(n_steps):
            psi, t = self.step(psi, t, f, **kwargs)

            if save_trajectory:
                trajectory.append(psi.clone())
                times.append(t)

        # Restore original dt
        self.dt = original_dt

        if save_trajectory:
            trajectory = torch.stack(trajectory)
            times = torch.tensor(times, device=self.device)
            return psi, trajectory, times
        else:
            return psi, None, None


def integrate_echozero(
    psi0: torch.Tensor,
    t_span: Tuple[float, float],
    I_t: torch.Tensor,
    desires: torch.Tensor,
    node_freqs: torch.Tensor,
    K: torch.Tensor,
    alpha: float = 0.10,
    beta: float = 0.05,
    lambda_hub: float = 0.02,
    dt: float = 0.01,
    method: str = "rk4",
    save_trajectory: bool = False,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Integrate EchoZero dynamics.

    Args:
        psi0: Initial state ψ(0)
        t_span: (t_start, t_end)
        I_t: Drive signal (constant or time-dependent)
        desires: Desire vectors
        node_freqs: Natural frequencies
        K: Coupling matrix
        alpha: Damping coefficient
        beta: Nonlinear damping
        lambda_hub: Hub coupling
        dt: Time step
        method: Integration method ('rk4' or 'euler')
        save_trajectory: Save full trajectory
        device: Computation device

    Returns:
        psi_final: Final state
        trajectory: Full trajectory (if requested)
        times: Time points (if requested)
    """
    from .dynamics import echozero_dynamics

    # Create derivative function with fixed parameters
    def f(psi, t, **kwargs):
        return echozero_dynamics(
            psi=psi,
            t=t,
            I_t=I_t,
            desires=desires,
            node_freqs=node_freqs,
            K=K,
            alpha=alpha,
            beta=beta,
            lambda_hub=lambda_hub,
        )

    if method == "rk4":
        solver = RK4Solver(dt=dt, device=device)
        return solver.integrate(
            psi0=psi0,
            t_span=t_span,
            f=f,
            save_trajectory=save_trajectory,
        )

    elif method == "euler":
        # Simple Euler method for comparison
        return integrate_euler(
            psi0=psi0,
            t_span=t_span,
            f=f,
            dt=dt,
            save_trajectory=save_trajectory,
            device=device,
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def integrate_euler(
    psi0: torch.Tensor,
    t_span: Tuple[float, float],
    f: Callable,
    dt: float = 0.01,
    save_trajectory: bool = False,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Simple Euler integration (for comparison/debugging).

    Args:
        psi0: Initial state
        t_span: Time interval
        f: Derivative function
        dt: Time step
        save_trajectory: Save trajectory
        device: Device

    Returns:
        psi_final, trajectory, times
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)

    psi = psi0.clone()
    t = t_start

    if save_trajectory:
        trajectory = [psi.clone()]
        times = [t]

    for _ in range(n_steps):
        dpsi_dt = f(psi, t)
        psi = psi + dpsi_dt * dt
        t = t + dt

        if save_trajectory:
            trajectory.append(psi.clone())
            times.append(t)

    if save_trajectory:
        trajectory = torch.stack(trajectory)
        times = torch.tensor(times, device=device)
        return psi, trajectory, times
    else:
        return psi, None, None


def adaptive_dt_control(
    error: torch.Tensor,
    dt_current: float,
    tolerance: float = 1e-4,
    safety_factor: float = 0.9,
    dt_min: float = 1e-6,
    dt_max: float = 0.1,
) -> float:
    """
    Adaptive time step control based on local error estimate.

    Args:
        error: Local truncation error estimate
        dt_current: Current time step
        tolerance: Error tolerance
        safety_factor: Safety factor for step size adjustment
        dt_min: Minimum allowed dt
        dt_max: Maximum allowed dt

    Returns:
        dt_new: New time step
    """
    error_norm = torch.abs(error).max().item()

    if error_norm < 1e-12:
        # Error very small, increase dt
        dt_new = min(dt_current * 2.0, dt_max)
    else:
        # Adjust based on error ratio
        ratio = (tolerance / error_norm) ** 0.25
        dt_new = dt_current * safety_factor * ratio
        dt_new = max(dt_min, min(dt_new, dt_max))

    return dt_new
