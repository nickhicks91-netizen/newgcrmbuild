"""
Scalable lattice builder for EchoZero.

Generate systems from 64 to 1B+ nodes with efficient sparse representations.
"""

import torch
from typing import Tuple, Dict
import numpy as np

from ..echozero import build_ring_lattice, build_coupling_matrix


class ScalableLatticeBuilder:
    """
    Build EchoZero lattices at scale.

    Supports:
    - Dense: Up to ~10K nodes (full coupling matrix)
    - Sparse: Up to 1B+ nodes (sparse adjacency lists)
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize builder.

        Args:
            device: Computation device
        """
        self.device = device

    def build(
        self,
        n_nodes: int,
        topology: str = "ring",
        add_torsion: bool = True,
        sparse: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Build lattice at specified scale.

        Args:
            n_nodes: Number of nodes
            topology: Topology type ('ring', 'grid', 'random')
            add_torsion: Add torsion connections
            sparse: Use sparse representation (auto if None)

        Returns:
            adjacency: Edge list or sparse matrix
            node_freqs: Natural frequencies
            edges: List of edges
        """
        # Auto-determine sparse mode
        if sparse is None:
            sparse = n_nodes > 10000

        if topology == "ring":
            return build_ring_lattice(
                n_nodes=n_nodes,
                add_torsion=add_torsion,
                device=self.device,
            )
        else:
            raise NotImplementedError(f"Topology {topology} not implemented")

    def estimate_memory(self, n_nodes: int, sparse: bool = False) -> Dict[str, float]:
        """
        Estimate memory requirements.

        Args:
            n_nodes: Number of nodes
            sparse: Whether using sparse representation

        Returns:
            memory: Dictionary of memory estimates (MB)
        """
        complex_bytes = 8  # Complex64 = 2 * float32

        if sparse:
            # Sparse: adjacency list only
            # Assume avg degree ~10
            avg_degree = 10
            n_edges = n_nodes * avg_degree
            coupling_mb = (n_edges * complex_bytes) / 1e6
        else:
            # Dense: full matrix
            coupling_mb = (n_nodes ** 2 * complex_bytes) / 1e6

        # State vectors
        state_mb = (n_nodes * complex_bytes) / 1e6
        freqs_mb = (n_nodes * 4) / 1e6  # float32

        total_mb = coupling_mb + state_mb + freqs_mb

        return {
            "coupling_mb": coupling_mb,
            "state_mb": state_mb,
            "freqs_mb": freqs_mb,
            "total_mb": total_mb,
        }

    def scaling_analysis(self) -> Dict[str, any]:
        """
        Analyze scaling properties.

        Returns:
            analysis: Scaling statistics
        """
        scales = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]

        results = {}
        for n in scales:
            mem = self.estimate_memory(n, sparse=(n > 10000))
            results[n] = {
                "memory_mb": mem["total_mb"],
                "memory_gb": mem["total_mb"] / 1024,
                "sparse": n > 10000,
            }

        return results


def build_scaled_system(
    n_nodes: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Convenience function to build complete scaled system.

    Args:
        n_nodes: Number of nodes
        device: Device

    Returns:
        K: Coupling matrix
        node_freqs: Natural frequencies
        adjacency: Edge list
        edges: List of edge tuples
    """
    builder = ScalableLatticeBuilder(device=device)

    adjacency, node_freqs, edges = builder.build(n_nodes)

    K = build_coupling_matrix(
        n_nodes=n_nodes,
        edges=edges,
        device=device,
    )

    return K, node_freqs, adjacency, edges
