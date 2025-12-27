"""
Lattice topology builder for EchoZero.

Implements ring lattice with optional torsion for complex network topologies.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List


class LatticeBuilder:
    """
    Builder for EchoZero lattice topologies.

    Supports:
    - Ring lattices (1D periodic)
    - Ring with torsion (adds N//8 long-range connections)
    - Custom topologies via adjacency specification
    """

    def __init__(self, n_nodes: int, device: str = "cpu"):
        """
        Initialize lattice builder.

        Args:
            n_nodes: Number of nodes in the lattice
            device: Computation device
        """
        self.n_nodes = n_nodes
        self.device = device

    def build_ring(
        self,
        coupling_strength: float = 1.0,
        add_torsion: bool = True,
        torsion_fraction: float = 0.125,  # N//8
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Build ring lattice with optional torsion.

        Ring topology: Each node connects to nearest neighbors (i-1, i+1)
        Torsion: Add N//8 evenly-spaced long-range connections

        Args:
            coupling_strength: Base coupling strength
            add_torsion: Whether to add torsion connections
            torsion_fraction: Fraction of nodes for torsion (default 1/8)

        Returns:
            adjacency: Adjacency list as tensor of edges
            edges: List of (source, target) tuples
        """
        edges = []

        # Ring connections: i -> (i+1) % N
        for i in range(self.n_nodes):
            next_node = (i + 1) % self.n_nodes
            edges.append((i, next_node))
            edges.append((next_node, i))  # Symmetric

        # Torsion connections: Add long-range links
        if add_torsion:
            n_torsion = max(1, int(self.n_nodes * torsion_fraction))
            step = self.n_nodes // n_torsion

            for i in range(0, self.n_nodes, step):
                target = (i + self.n_nodes // 2) % self.n_nodes
                edges.append((i, target))
                edges.append((target, i))  # Symmetric

        # Remove duplicates
        edges = list(set(edges))

        # Convert to tensor
        adjacency = torch.tensor(edges, dtype=torch.long, device=self.device)

        return adjacency, edges

    def build_frequencies(
        self,
        base_freq: float = 1.0,
        freq_std: float = 0.1,
        mode: str = "random",
    ) -> torch.Tensor:
        """
        Generate natural frequencies for each node.

        Args:
            base_freq: Base frequency (mean)
            freq_std: Standard deviation
            mode: Distribution mode ('random', 'uniform', 'linspace')

        Returns:
            node_freqs: Natural frequencies ω ∈ ℝ^N
        """
        if mode == "random":
            # Gaussian distribution
            freqs = torch.randn(self.n_nodes, device=self.device) * freq_std + base_freq

        elif mode == "uniform":
            # Uniform distribution
            freqs = (
                torch.rand(self.n_nodes, device=self.device) * 2 * freq_std
                + (base_freq - freq_std)
            )

        elif mode == "linspace":
            # Linearly spaced
            freqs = torch.linspace(
                base_freq - freq_std,
                base_freq + freq_std,
                self.n_nodes,
                device=self.device,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return freqs

    def visualize_topology(self, edges: List[Tuple[int, int]]) -> str:
        """
        Generate ASCII visualization of lattice topology.

        Args:
            edges: List of (source, target) edge tuples

        Returns:
            viz: String representation
        """
        viz = f"Lattice Topology (N={self.n_nodes})\n"
        viz += f"Total edges: {len(edges)}\n"
        viz += f"Avg degree: {2 * len(edges) / self.n_nodes:.2f}\n"

        # Count connections per node
        degree = [0] * self.n_nodes
        for src, tgt in edges:
            degree[src] += 1

        viz += f"Degree distribution:\n"
        viz += f"  Min: {min(degree)}\n"
        viz += f"  Max: {max(degree)}\n"
        viz += f"  Mean: {np.mean(degree):.2f}\n"

        return viz


def build_ring_lattice(
    n_nodes: int,
    coupling_strength: float = 1.0,
    add_torsion: bool = True,
    base_freq: float = 1.0,
    freq_std: float = 0.1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Convenience function to build complete ring lattice.

    Args:
        n_nodes: Number of oscillator nodes
        coupling_strength: Coupling strength
        add_torsion: Add torsion connections
        base_freq: Base frequency
        freq_std: Frequency standard deviation
        device: Computation device

    Returns:
        adjacency: Edge list tensor
        node_freqs: Natural frequencies
        edges: List of edge tuples
    """
    builder = LatticeBuilder(n_nodes, device)

    adjacency, edges = builder.build_ring(
        coupling_strength=coupling_strength,
        add_torsion=add_torsion,
    )

    node_freqs = builder.build_frequencies(
        base_freq=base_freq,
        freq_std=freq_std,
        mode="random",
    )

    return adjacency, node_freqs, edges


def analyze_topology(edges: List[Tuple[int, int]], n_nodes: int) -> dict:
    """
    Analyze topology properties.

    Args:
        edges: List of edge tuples
        n_nodes: Number of nodes

    Returns:
        stats: Dictionary of topology statistics
    """
    # Compute degree distribution
    degree = [0] * n_nodes
    for src, tgt in edges:
        degree[src] += 1

    # Clustering coefficient (approximate)
    # For ring: perfect clustering in local neighborhoods
    # With torsion: reduced clustering

    stats = {
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "avg_degree": 2 * len(edges) / n_nodes,
        "min_degree": min(degree),
        "max_degree": max(degree),
        "degree_std": np.std(degree),
    }

    return stats
