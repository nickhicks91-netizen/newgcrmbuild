#!/usr/bin/env python3
"""
EchoZero State Exporter for 3D Visualization Demo

This script runs the real EchoZero + Spiral Lattice + Möbius system
and exports state snapshots to JSON for browser-based visualization.

Usage:
    python export_echozero_state.py
    # Opens browser demo automatically
    # Updates state.json in real-time
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import math
import numpy as np
from pathlib import Path

# Import real EchoZero components
try:
    from grcm.echozero.spiral import SpiralLattice
    from grcm.echozero.mobius import MobiusEchoLayer
    PYTORCH_AVAILABLE = True
    import torch
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - using theoretical simulation")


class EchoZeroVisualizer:
    """Real-time EchoZero state exporter for 3D visualization."""

    def __init__(self, hidden_dim=32, memory_length=256):
        self.hidden_dim = hidden_dim
        self.memory_length = memory_length
        self.step_count = 0

        if PYTORCH_AVAILABLE:
            # Real PyTorch implementation
            self.spiral = SpiralLattice(
                hidden_dim=hidden_dim,
                memory_length=memory_length,
                radial_decay=0.015,
                angular_velocity=0.45,
                stability_gain=1.6,
            )
            self.mobius = MobiusEchoLayer(
                hidden_dim=hidden_dim,
                loop_len=memory_length,
                torsion_gain=3.0,
                damping_gain=12.0,
            )
        else:
            # Theoretical simulation
            self.spiral_state = {"radius": 1.0, "angle": 0.0, "coherence": 0.5}
            self.mobius_state = {"energy": 0.1, "gate": 0.9}

    def step(self):
        """Run one step of EchoZero dynamics."""
        self.step_count += 1

        if PYTORCH_AVAILABLE:
            # Real implementation
            x = torch.randn(1, self.hidden_dim) * 0.1

            # Spiral Lattice
            x_spiral, spiral_metrics = self.spiral(x)

            # Möbius Echo Layer
            x_mobius, mobius_metrics = self.mobius(x_spiral)

            return {
                "spiral": {
                    "coherence": spiral_metrics["coherence"],
                    "radius": spiral_metrics["radius"],
                    "angle": spiral_metrics["angle"],
                    "stabilizer": spiral_metrics["stabilizer"],
                },
                "mobius": {
                    "energy": mobius_metrics["energy"],
                    "gate_mean": mobius_metrics["gate_mean"],
                    "gate_min": mobius_metrics["gate_min"],
                    "gate_max": mobius_metrics["gate_max"],
                    "purity": mobius_metrics["purity"],
                },
                "state_vector": x_mobius[0].tolist(),
            }
        else:
            # Theoretical simulation
            t = self.step_count
            radius = math.exp(-0.015 * (t % self.memory_length))
            angle = 0.45 * t
            coherence = 0.5 + 0.3 * math.sin(t * 0.1)
            energy = 0.05 + 0.03 * math.sin(t * 0.15)
            gate = 1.0 - energy

            return {
                "spiral": {
                    "coherence": coherence,
                    "radius": radius,
                    "angle": angle,
                    "stabilizer": 1.0 + 1.6 * coherence,
                },
                "mobius": {
                    "energy": energy,
                    "gate_mean": gate,
                    "gate_min": gate * 0.8,
                    "gate_max": min(1.0, gate * 1.2),
                    "purity": 0.95,
                },
                "state_vector": [math.sin(angle + i * 0.1) * radius for i in range(32)],
            }

    def export_state(self, output_path="demo/state.json"):
        """Export current state to JSON."""
        metrics = self.step()

        # Compute derived values for visualization
        spiral_metrics = metrics["spiral"]
        mobius_metrics = metrics["mobius"]

        # Spiral lattice position (3D coordinates)
        r = spiral_metrics["radius"]
        theta = spiral_metrics["angle"]
        t_norm = (self.step_count % self.memory_length) / self.memory_length

        position = {
            "x": r * math.cos(theta) * 10.0,
            "y": (t_norm - 0.5) * 20.0,  # Vertical spread
            "z": r * math.sin(theta) * 10.0,
        }

        # Color based on coherence
        coherence = spiral_metrics["coherence"]
        if coherence > 0.7:
            color = "healthy"  # Green
        elif coherence > 0.4:
            color = "moderate"  # Yellow-green
        else:
            color = "stressed"  # Orange-red

        # Torsion from Möbius energy
        torsion = mobius_metrics["energy"]

        state = {
            "step": self.step_count,
            "timestamp": time.time(),
            "position": position,
            "spiral": spiral_metrics,
            "mobius": mobius_metrics,
            "color": color,
            "torsion": torsion,
            "coherence": coherence,
            "mode": "pytorch" if PYTORCH_AVAILABLE else "theoretical",
        }

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(state, f, indent=2)

        return state


def run_continuous(fps=30, duration=None):
    """Run continuous export loop."""
    viz = EchoZeroVisualizer()
    frame_time = 1.0 / fps

    print(f"🌿 EchoZero State Exporter")
    print(f"Mode: {'PyTorch (Real)' if PYTORCH_AVAILABLE else 'Theoretical Simulation'}")
    print(f"FPS: {fps}")
    print(f"Output: demo/state.json")
    print()
    print("Press Ctrl+C to stop")
    print()

    start_time = time.time()
    frames = 0

    try:
        while True:
            frame_start = time.time()

            # Export state
            state = viz.export_state()

            frames += 1
            if frames % 30 == 0:
                elapsed = time.time() - start_time
                actual_fps = frames / elapsed
                print(f"Frame {frames:5d} | "
                      f"Coherence: {state['coherence']:.3f} | "
                      f"Torsion: {state['torsion']:.4f} | "
                      f"Radius: {state['spiral']['radius']:.4f} | "
                      f"FPS: {actual_fps:.1f}")

            # Check duration limit
            if duration and (time.time() - start_time) >= duration:
                break

            # Sleep to maintain target FPS
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n✅ Export stopped")
        print(f"Total frames: {frames}")
        print(f"Duration: {time.time() - start_time:.1f}s")


def export_single_snapshot():
    """Export a single state snapshot."""
    viz = EchoZeroVisualizer()
    state = viz.export_state()

    print("✅ Single snapshot exported to demo/state.json")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EchoZero 3D Visualization State Exporter")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: infinite)")
    parser.add_argument("--single", action="store_true", help="Export single snapshot and exit")

    args = parser.parse_args()

    if args.single:
        export_single_snapshot()
    else:
        run_continuous(fps=args.fps, duration=args.duration)
