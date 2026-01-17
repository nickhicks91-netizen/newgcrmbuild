"""
Hopfield-Guided Spatial Compression

Instead of blind PCA compression, this decoder:
- Uses eigenvectors of the Hopfield weight matrix as compression basis
- Leverages learned attractor structure for better reconstruction
- Expected 20-40% error vs. 60-80% for blind PCA

Theory:
- Hopfield attractors span a learned subspace
- Top eigenvectors of W capture this subspace
- Projecting onto this basis preserves attractor-relevant information
"""

import numpy as np
from typing import Optional, List


class HopfieldSpatialDecoder:
    """
    Spatial decoder that uses Hopfield attractor structure for compression.

    Instead of generic PCA (which ignores learned structure), this uses
    the eigenbasis of the Hopfield weight matrix as the compression basis.
    """

    def __init__(self, dim: int = 64, rank: int = 16):
        """
        Args:
            dim: Full state dimensionality
            rank: Compression rank (number of eigenvectors to use)
        """
        self.dim = dim
        self.rank = rank
        self.proj = None  # Projection matrix (dim, rank)
        self.inv_proj = None  # Inverse projection (rank, dim)
        self.codes = []  # Stored compressed codes
        self.fitted = False

    def fit_projection(self, hopfield) -> None:
        """
        Fit compression basis using Hopfield weight matrix eigenvectors.

        Args:
            hopfield: Hopfield network with weight matrix W (dim, dim)
        """
        W = hopfield.W  # (dim, dim)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(W)

        # Top-k eigenvectors capture strongest attractor structure
        # eigh returns eigenvalues in ascending order, so take last k
        top_k_indices = np.argsort(np.abs(eigenvalues))[-self.rank:]

        # Projection basis: top-k eigenvectors
        self.proj = eigenvectors[:, top_k_indices]  # (dim, rank)
        self.inv_proj = self.proj.T  # (rank, dim)

        self.fitted = True

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Encode state into compressed representation.

        Args:
            state: State vector (dim,)

        Returns:
            Compressed code (rank,)
        """
        if not self.fitted:
            raise RuntimeError("Must call fit_projection() before encoding")

        # Project onto attractor eigenbasis
        code = self.inv_proj @ state  # (rank,)
        return code.astype(np.float32)

    def decode(self, code: np.ndarray) -> np.ndarray:
        """
        Decode compressed representation back to state space.

        Args:
            code: Compressed code (rank,)

        Returns:
            Reconstructed state (dim,)
        """
        if not self.fitted:
            raise RuntimeError("Must call fit_projection() before decoding")

        # Project from attractor eigenbasis back to state space
        state = self.proj @ code  # (dim,)
        return state.astype(np.float32)

    def store_attractor(self, state: np.ndarray) -> int:
        """
        Store attractor state in compressed form.

        Args:
            state: State vector (dim,)

        Returns:
            Index of stored code
        """
        code = self.encode(state)
        self.codes.append(code)
        return len(self.codes) - 1

    def reconstruct_attractor(self, idx: int) -> np.ndarray:
        """
        Reconstruct attractor from stored code.

        Args:
            idx: Index of stored code

        Returns:
            Reconstructed state (dim,)
        """
        if idx < 0 or idx >= len(self.codes):
            raise IndexError(f"Code index {idx} out of range [0, {len(self.codes)})")

        code = self.codes[idx]
        return self.decode(code)

    def encode_decode_cycle(self, state: np.ndarray) -> np.ndarray:
        """
        Full encode-decode cycle (for testing reconstruction quality).

        Args:
            state: Original state (dim,)

        Returns:
            Reconstructed state (dim,)
        """
        code = self.encode(state)
        return self.decode(code)

    def reconstruction_error(self, state: np.ndarray) -> float:
        """
        Compute reconstruction error for a state.

        Args:
            state: Original state (dim,)

        Returns:
            Relative reconstruction error
        """
        reconstructed = self.encode_decode_cycle(state)
        error = np.linalg.norm(state - reconstructed) / (np.linalg.norm(state) + 1e-8)
        return float(error)

    def compression_ratio(self) -> float:
        """Compute compression ratio."""
        return self.dim / self.rank

    def memory_usage_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        proj_bytes = self.dim * self.rank * 4  # float32
        inv_proj_bytes = self.rank * self.dim * 4
        codes_bytes = len(self.codes) * self.rank * 4
        return proj_bytes + inv_proj_bytes + codes_bytes

    def stats(self) -> dict:
        """Get decoder statistics."""
        return {
            "dim": self.dim,
            "rank": self.rank,
            "compression_ratio": self.compression_ratio(),
            "fitted": self.fitted,
            "num_codes": len(self.codes),
            "memory_bytes": self.memory_usage_bytes(),
            "memory_kb": self.memory_usage_bytes() / 1024,
        }


class HopfieldSpatialDecoderWithSnapback:
    """
    Enhanced decoder that projects reconstructions back onto Hopfield attractors.

    This combines:
    1. Hopfield eigenbasis compression (better than PCA)
    2. Attractor snapback (project onto nearest attractor)

    Expected reconstruction error: 10-20% (vs. 20-40% without snapback)
    """

    def __init__(self, dim: int = 64, rank: int = 16, hopfield=None):
        """
        Args:
            dim: Full state dimensionality
            rank: Compression rank
            hopfield: Hopfield network (optional, for snapback)
        """
        self.decoder = HopfieldSpatialDecoder(dim=dim, rank=rank)
        self.hopfield = hopfield
        self.dim = dim
        self.rank = rank

    def fit_projection(self, hopfield) -> None:
        """Fit compression basis and store Hopfield network."""
        self.hopfield = hopfield
        self.decoder.fit_projection(hopfield)

    def encode(self, state: np.ndarray) -> np.ndarray:
        """Encode state."""
        return self.decoder.encode(state)

    def decode(self, code: np.ndarray, snapback: bool = True) -> np.ndarray:
        """
        Decode with optional attractor snapback.

        Args:
            code: Compressed code (rank,)
            snapback: Whether to project onto nearest attractor

        Returns:
            Reconstructed state (dim,)
        """
        state = self.decoder.decode(code)

        if snapback and self.hopfield is not None:
            # Run one Hopfield step to snap to nearest attractor
            state = self.hopfield.step(state)

        return state

    def store_attractor(self, state: np.ndarray) -> int:
        """Store attractor."""
        return self.decoder.store_attractor(state)

    def reconstruct_attractor(self, idx: int, snapback: bool = True) -> np.ndarray:
        """Reconstruct attractor with optional snapback."""
        code = self.decoder.codes[idx]
        return self.decode(code, snapback=snapback)

    def reconstruction_error(self, state: np.ndarray, snapback: bool = True) -> float:
        """Compute reconstruction error with optional snapback."""
        code = self.encode(state)
        reconstructed = self.decode(code, snapback=snapback)
        error = np.linalg.norm(state - reconstructed) / (np.linalg.norm(state) + 1e-8)
        return float(error)

    @property
    def fitted(self) -> bool:
        return self.decoder.fitted

    def stats(self) -> dict:
        """Get decoder statistics."""
        return self.decoder.stats()
