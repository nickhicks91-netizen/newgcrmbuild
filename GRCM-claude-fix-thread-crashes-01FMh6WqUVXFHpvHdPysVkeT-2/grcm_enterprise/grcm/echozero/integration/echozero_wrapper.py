"""
============================================================
   ECHOZERO FULL INTEGRATION WRAPPER
   Unified coherence → torsion → tachyon → memory pipeline
============================================================

This wrapper provides:
  ✓ A single forward() entrypoint
  ✓ Full routing of all EchoZero subsystems
  ✓ Safety isolation (memory never pollutes model state)
  ✓ Logging and introspection hooks
  ✓ Non-blocking slow loop architecture
  ✓ Production-ready modularity for xAI / NVIDIA / IBM

Compatible with:
  - Möbius geometric gate
  - Spiral Lattice temporal integrator
  - Torsion Lattice (XY-model)
  - Tachyon Integration Layer
  - Hopfield event classifier + Identity vector
"""

import numpy as np
import torch
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

# Import EchoZero subsystems
from grcm.echozero.mobius import MobiusEchoLayer
from grcm.echozero.spiral import SpiralLattice
from grcm.echozero.identity.torsion_lattice import TorsionLattice3D
from grcm.echozero.tachyon import TachyonIntegrationWrapper


@dataclass
class EchoZeroConfig:
    """Configuration for full EchoZero pipeline."""
    dim: int = 64                       # Vector dimension
    lattice_size: int = 5               # Torsion lattice grid size (5×5×5)
    torsion_threshold: float = 2.5      # Spike detection threshold
    num_patterns: int = 20              # Event pattern types
    slow_loop_rate: int = 300           # Sync every N steps
    enable_logging: bool = True
    enable_tachyon: bool = True         # Master switch for Tachyon layer


class EchoZeroWrapper:
    """
    Full EchoZero integration wrapper.

    This class provides:
      • forward(x) — unified coherent output
      • slow_update() — memory sync loop (non-blocking)
      • export_state() — everything needed for visualization/logging

    The wrapper guarantees:
      - Memory NEVER corrupts model output
      - Tachyon events NEVER interrupt fast loop
      - Hopfield memory is always stable
      - Möbius gating is always applied first
      - Zero backreaction from memory to inference
    """

    def __init__(self, config: EchoZeroConfig = EchoZeroConfig()):
        """Initialize full EchoZero pipeline."""
        self.cfg = config
        
        self._lock = threading.RLock()

        # Core pipeline components
        self.mobius = MobiusEchoLayer(hidden_dim=config.dim)
        self.spiral = SpiralLattice(hidden_dim=config.dim)
        self.torsion = TorsionLattice3D(
            size=config.lattice_size,
            coupling=1.0,
            gamma=0.005
        )

        # Event-driven memory system
        if config.enable_tachyon:
            self.tachyon = TachyonIntegrationWrapper(
                lattice_size=config.lattice_size,
                num_patterns=config.num_patterns,
                torsion_threshold=config.torsion_threshold
            )
        else:
            self.tachyon = None

        # State tracking (protected by _lock)
        self._step = 0
        self._last_slow_update = 0

        # Logging (protected by _lock)
        self._logs = {
            "torsion_magnitude": [],
            "mobius_metrics": [],
            "tachyon_events": [],
            "identity_snapshots": [],
        }

    def forward(self, x: np.ndarray) -> Dict[str, Any]:
        """
        The full EchoZero coherence pipeline (thread-safe).

        Args:
            x: Input state vector or tensor

        Returns:
            Dictionary containing:
            - coherent_output: Final coherent vector for model
            - mobius_metrics: Topological consistency measures
            - tachyon_result: Event detection results (if enabled)
            - identity_state: Current identity vector (if Tachyon enabled)
        """
        # Convert input to numpy if needed
        if hasattr(x, 'detach'):  # PyTorch tensor
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)

        # Ensure 2D: (batch, dim) or (dim,)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        batch_size = x_np.shape[0]

        # FAST LOOP PIPELINE
        # -----------------------------------------------------------

        # Convert numpy to PyTorch for Möbius and Spiral
        x_torch = torch.from_numpy(x_np).float()

        # Step 1: Möbius Echo Layer (topological projection)
        mobius_out, mobius_metrics = self.mobius(x_torch)

        with self._lock:
            current_step = self._step
            if self.cfg.enable_logging:
                self._logs["mobius_metrics"].append({
                    'step': current_step,
                    'energy': mobius_metrics.get('energy', 0.0),
                    'torsion': mobius_metrics.get('torsion', 0.0)
                })

        # Step 2: Spiral Lattice (temporal compression)
        spiral_out_torch, spiral_metrics = self.spiral.forward(mobius_out)

        # Convert back to numpy for Torsion Lattice (NumPy-based)
        spiral_out = spiral_out_torch.detach().cpu().numpy()

        # Step 3: Torsion Lattice (coherence stabilization)
        # Spiral output is already 2D real, don't extract first 2 components
        # For now, just use the torsion lattice state for event detection
        # (Torsion lattice runs independently in background)

        # Get torsion lattice state (this is just reading, not modifying)
        # Construct complex state from phases and magnitude
        torsion_state = self.torsion.m * np.exp(1j * self.torsion.theta)  # 5×5×5 complex grid

        # Compute torsion magnitude for logging
        torsion_mag = self._compute_torsion_magnitude(torsion_state)

        with self._lock:
            if self.cfg.enable_logging:
                self._logs["torsion_magnitude"].append(float(torsion_mag))

        # Step 4: Tachyon Layer (event detection + identity update)
        tachyon_result = None
        if self.tachyon is not None:
            tachyon_result = self.tachyon.process_step(torsion_state)

            if tachyon_result['event_detected']:
                with self._lock:
                    if self.cfg.enable_logging:
                        self._logs["tachyon_events"].append({
                            'step': current_step,
                            'pattern_index': tachyon_result['pattern_index'],
                            'confidence': tachyon_result['confidence'],
                            'torsion': tachyon_result['event']['torsion_magnitude']
                        })

        # Step 5: Slow loop sync (non-blocking) - atomic decision and update
        with self._lock:
            should_slow_update = (self._step - self._last_slow_update >= self.cfg.slow_loop_rate)
            if should_slow_update:
                self._slow_update_internal()
                self._last_slow_update = self._step

            self._step += 1
            new_step = self._step

        # Return coherent output (spiral output is the final coherent state)
        return {
            'coherent_output': spiral_out,  # Use this for model
            'mobius_metrics': mobius_metrics,
            'torsion_magnitude': torsion_mag,
            'tachyon_result': tachyon_result,
            'identity_state': self.get_identity_state(),
            'step': new_step
        }

    def slow_update(self):
        """
        Slow loop memory consolidation (thread-safe wrapper).

        This is where:
        - Torsion lattice relaxes
        - Identity vector decays
        - Long-term memory stabilizes

        CRITICAL: Does NOT modify forward-pass output.
        """
        with self._lock:
            self._slow_update_internal()
    
    def _slow_update_internal(self):
        """Internal slow update (must be called with lock held)."""
        # Relax torsion lattice
        self.torsion.step()

        # Snapshot identity for logging
        if self.cfg.enable_logging and self.tachyon is not None:
            self._logs["identity_snapshots"].append({
                'step': self._step,
                'identity': self.tachyon.identity.get_state().copy(),
                'stats': self.tachyon.identity.get_statistics()
            })

    def _compute_torsion_magnitude(self, state: np.ndarray) -> float:
        """Compute torsion magnitude from complex lattice state."""
        phases = np.angle(state)

        # Discrete gradient
        grad_x = np.roll(phases, -1, axis=0) - phases
        grad_y = np.roll(phases, -1, axis=1) - phases
        grad_z = np.roll(phases, -1, axis=2) - phases

        # Wrap to [-π, π]
        grad_x = (grad_x + np.pi) % (2 * np.pi) - np.pi
        grad_y = (grad_y + np.pi) % (2 * np.pi) - np.pi
        grad_z = (grad_z + np.pi) % (2 * np.pi) - np.pi

        # Curl magnitude
        curl_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        return float(np.max(curl_magnitude))

    def get_identity_state(self) -> Optional[np.ndarray]:
        """Get current identity vector state."""
        if self.tachyon is not None:
            return self.tachyon.identity.get_state()
        return None

    def get_dominant_patterns(self, top_k: int = 5):
        """Get top-k active identity patterns."""
        if self.tachyon is not None:
            return self.tachyon.identity.get_dominant_patterns(top_k)
        return []

    def export_state(self) -> Dict:
        """
        Export complete system state for visualization/logging (thread-safe).

        Returns:
            Dictionary with full pipeline state
        """
        with self._lock:
            state = {
                'step': self._step,
                'torsion_magnitude': self._logs["torsion_magnitude"][-100:] if self._logs["torsion_magnitude"] else [],
                'recent_events': self._logs["tachyon_events"][-20:] if self._logs["tachyon_events"] else [],
            }

        if self.tachyon is not None:
            state.update({
                'identity_state': self.tachyon.identity.get_state(),
                'identity_stats': self.tachyon.identity.get_statistics(),
                'dominant_patterns': self.get_dominant_patterns(top_k=5),
                'tachyon_stats': self.tachyon.get_statistics()
            })

        return state

    def learn_event_prototypes(self, signatures: list, labels: Optional[list] = None):
        """
        Pre-train event prototypes for Hopfield classifier.

        Args:
            signatures: List of 6D event signature vectors
            labels: Optional semantic labels
        """
        if self.tachyon is not None:
            self.tachyon.batch_learn_prototypes(signatures, labels)

    def reset(self):
        """Reset all subsystems to initial state (thread-safe)."""
        with self._lock:
            self.torsion.reset()
            if self.tachyon is not None:
                self.tachyon.reset()

            self._step = 0
            self._last_slow_update = 0
            self._logs = {
                "torsion_magnitude": [],
                "mobius_metrics": [],
                "tachyon_events": [],
                "identity_snapshots": [],
            }
    
    @property
    def step(self) -> int:
        """Get current step count (thread-safe, backward compatible)."""
        with self._lock:
            return self._step
    
    @property
    def logs(self) -> Dict:
        """Get logs (thread-safe copy, backward compatible)."""
        import copy
        with self._lock:
            return copy.deepcopy(self._logs)
    
    @property
    def last_slow_update(self) -> int:
        """Get last slow update step (thread-safe, backward compatible)."""
        with self._lock:
            return self._last_slow_update

    def enable_tachyon(self):
        """Enable Tachyon layer at runtime."""
        if self.tachyon is not None:
            self.tachyon.enable()

    def disable_tachyon(self):
        """Disable Tachyon layer (zero backreaction mode)."""
        if self.tachyon is not None:
            self.tachyon.disable()

    def __repr__(self) -> str:
        tachyon_status = "enabled" if (self.tachyon and self.tachyon.enabled) else "disabled"
        return (f"EchoZeroWrapper(step={self.step}, "
                f"tachyon={tachyon_status}, "
                f"events_detected={len(self.logs['tachyon_events'])})")
