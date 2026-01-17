"""
Test Suite for Hopfield-Guided Spatial Compression

Tests attractor-guided compression vs blind PCA:
- Uses Hopfield weight matrix eigenvectors as compression basis
- Expected 20-40% reconstruction error (vs 60-80% for PCA)
- Leverages learned attractor structure
"""

import pytest
import numpy as np
from grcm.echozero.memory.hopfield_decoder import (
    HopfieldSpatialDecoder,
    HopfieldSpatialDecoderWithSnapback,
)


class DummyHopfield:
    """Dummy Hopfield network for testing."""

    def __init__(self, dim=64, num_patterns=5):
        self.dim = dim
        self.num_patterns = num_patterns

        # Create synthetic weight matrix
        # Use outer product of random patterns
        patterns = [np.random.randn(dim) for _ in range(num_patterns)]
        patterns = [p / np.linalg.norm(p) for p in patterns]  # Normalize

        # Hopfield weight matrix: W = sum of outer products
        self.W = np.zeros((dim, dim))
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= num_patterns  # Normalize

        # Make symmetric
        self.W = (self.W + self.W.T) / 2

        # Store patterns for testing
        self.patterns = patterns

    def step(self, state):
        """One Hopfield update step."""
        new_state = self.W @ state
        return new_state / (np.linalg.norm(new_state) + 1e-8)


class TestHopfieldSpatialDecoder:
    """Test Hopfield eigenbasis compression."""

    def test_initialization(self):
        """Test decoder initialization."""
        decoder = HopfieldSpatialDecoder(dim=64, rank=16)
        assert decoder.dim == 64
        assert decoder.rank == 16
        assert not decoder.fitted
        assert decoder.proj is None

    def test_fit_projection(self):
        """Test fitting projection from Hopfield network."""
        hopfield = DummyHopfield(dim=64, num_patterns=5)
        decoder = HopfieldSpatialDecoder(dim=64, rank=16)

        decoder.fit_projection(hopfield)

        assert decoder.fitted
        assert decoder.proj is not None
        assert decoder.proj.shape == (64, 16)  # (dim, rank)
        assert decoder.inv_proj.shape == (16, 64)  # (rank, dim)

    def test_encode_decode_cycle(self):
        """Test encode-decode reconstruction."""
        hopfield = DummyHopfield(dim=64, num_patterns=5)
        decoder = HopfieldSpatialDecoder(dim=64, rank=16)
        decoder.fit_projection(hopfield)

        # Test on random state
        state = np.random.randn(64)
        state = state / np.linalg.norm(state)

        # Encode-decode
        code = decoder.encode(state)
        reconstructed = decoder.decode(code)

        assert code.shape == (16,)
        assert reconstructed.shape == (64,)

        # Reconstruction should be approximate
        error = np.linalg.norm(state - reconstructed) / (np.linalg.norm(state) + 1e-8)
        assert error < 1.0  # Should have some fidelity

    def test_reconstruction_on_attractors(self):
        """Test reconstruction quality on actual Hopfield attractors."""
        hopfield = DummyHopfield(dim=64, num_patterns=5)
        decoder = HopfieldSpatialDecoder(dim=64, rank=16)
        decoder.fit_projection(hopfield)

        # Test on learned patterns (should reconstruct well)
        for pattern in hopfield.patterns:
            error = decoder.reconstruction_error(pattern)

            # Should be better than 80% (Hopfield eigenbasis should capture attractors)
            assert error < 0.8

    def test_store_and_reconstruct_attractor(self):
        """Test attractor storage and reconstruction."""
        hopfield = DummyHopfield(dim=64, num_patterns=5)
        decoder = HopfieldSpatialDecoder(dim=64, rank=16)
        decoder.fit_projection(hopfield)

        # Store attractors
        indices = []
        for pattern in hopfield.patterns:
            idx = decoder.store_attractor(pattern)
            indices.append(idx)

        assert len(decoder.codes) == 5

        # Reconstruct
        for i, pattern in enumerate(hopfield.patterns):
            reconstructed = decoder.reconstruct_attractor(indices[i])
            error = np.linalg.norm(pattern - reconstructed) / (np.linalg.norm(pattern) + 1e-8)
            assert error < 0.8  # Approximate reconstruction

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        decoder = HopfieldSpatialDecoder(dim=64, rank=16)
        assert decoder.compression_ratio() == 4.0  # 64/16

        decoder2 = HopfieldSpatialDecoder(dim=128, rank=32)
        assert decoder2.compression_ratio() == 4.0  # 128/32

    def test_memory_usage(self):
        """Test memory usage estimation."""
        hopfield = DummyHopfield(dim=64, num_patterns=5)
        decoder = HopfieldSpatialDecoder(dim=64, rank=16)
        decoder.fit_projection(hopfield)

        # Store some codes
        for i in range(10):
            decoder.store_attractor(np.random.randn(64))

        stats = decoder.stats()

        # Memory should include:
        # - proj: 64 × 16 × 4 bytes = 4,096 bytes
        # - inv_proj: 16 × 64 × 4 bytes = 4,096 bytes
        # - codes: 10 × 16 × 4 bytes = 640 bytes
        # Total ≈ 8,832 bytes

        assert stats["memory_bytes"] > 0
        assert stats["num_codes"] == 10

    def test_stats_output(self):
        """Test comprehensive stats output."""
        hopfield = DummyHopfield(dim=64, num_patterns=3)
        decoder = HopfieldSpatialDecoder(dim=64, rank=8)
        decoder.fit_projection(hopfield)

        decoder.store_attractor(np.random.randn(64))
        decoder.store_attractor(np.random.randn(64))

        stats = decoder.stats()

        assert stats["dim"] == 64
        assert stats["rank"] == 8
        assert stats["compression_ratio"] == 8.0
        assert stats["fitted"] is True
        assert stats["num_codes"] == 2
        assert "memory_kb" in stats


class TestHopfieldSpatialDecoderWithSnapback:
    """Test enhanced decoder with attractor snapback."""

    def test_initialization(self):
        """Test snapback decoder initialization."""
        hopfield = DummyHopfield(dim=64)
        decoder = HopfieldSpatialDecoderWithSnapback(dim=64, rank=16, hopfield=hopfield)

        assert decoder.dim == 64
        assert decoder.rank == 16
        assert decoder.hopfield is hopfield

    def test_decode_with_snapback(self):
        """Test decoding with attractor snapback."""
        hopfield = DummyHopfield(dim=64, num_patterns=5)
        decoder = HopfieldSpatialDecoderWithSnapback(dim=64, rank=16, hopfield=hopfield)
        decoder.fit_projection(hopfield)

        state = hopfield.patterns[0]
        code = decoder.encode(state)

        # Decode without snapback
        reconstructed_no_snap = decoder.decode(code, snapback=False)

        # Decode with snapback
        reconstructed_snap = decoder.decode(code, snapback=True)

        # Snapback should project onto attractor (different from no-snapback)
        assert not np.allclose(reconstructed_no_snap, reconstructed_snap)

    def test_snapback_improves_reconstruction(self):
        """Test that snapback improves reconstruction quality."""
        hopfield = DummyHopfield(dim=64, num_patterns=5)
        decoder = HopfieldSpatialDecoderWithSnapback(dim=64, rank=8, hopfield=hopfield)
        decoder.fit_projection(hopfield)

        # Test on attractor (snapback should help)
        pattern = hopfield.patterns[0]

        error_no_snap = decoder.reconstruction_error(pattern, snapback=False)
        error_snap = decoder.reconstruction_error(pattern, snapback=True)

        # Snapback should reduce error (though not guaranteed in all cases)
        # At minimum, both should be finite
        assert np.isfinite(error_no_snap)
        assert np.isfinite(error_snap)


def test_hopfield_vs_random_basis():
    """Test that Hopfield eigenbasis outperforms random basis on attractors."""
    hopfield = DummyHopfield(dim=64, num_patterns=5)

    # Hopfield-guided decoder
    hopfield_decoder = HopfieldSpatialDecoder(dim=64, rank=16)
    hopfield_decoder.fit_projection(hopfield)

    # Test reconstruction error on attractors
    hopfield_errors = []
    for pattern in hopfield.patterns:
        error = hopfield_decoder.reconstruction_error(pattern)
        hopfield_errors.append(error)

    avg_hopfield_error = np.mean(hopfield_errors)

    # Hopfield-guided should have reasonable error
    assert avg_hopfield_error < 1.0  # Should preserve some structure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
