"""
Pytest configuration and shared fixtures
"""
import pytest
import torch
import tempfile
from pathlib import Path
import yaml

# Import GRCM components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm import (
    ModularGRCM,
    GRCMConfig,
    load_config,
    GRCMOptimizer,
    GRCMBenchmark,
    BenchmarkConfig
)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_config():
    """Default GRCM configuration"""
    return GRCMConfig()


@pytest.fixture
def small_config():
    """Smaller configuration for faster tests"""
    return GRCMConfig(
        input_dim=10,
        freq_dim=4,
        memory_size=16
    )


@pytest.fixture
def temp_config_file(tmp_path):
    """Temporary YAML config file"""
    config_dict = {
        'input_dim': 10,
        'freq_dim': 4,
        'memory_size': 16,
        'attention': {'base_bandwidth': 0.5},
        'desire': {'num_desires': 2},
        'device': 'cpu'
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)

    return str(config_path)


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def grcm_model(small_config):
    """GRCM model with small config"""
    model = ModularGRCM(small_config)
    model.eval()
    return model


@pytest.fixture
def grcm_model_default():
    """GRCM model with default config"""
    model = ModularGRCM()
    model.eval()
    return model


# ============================================================================
# Input Fixtures
# ============================================================================

@pytest.fixture
def dummy_inputs_small():
    """Small dummy inputs for testing"""
    return {
        'image_emb': torch.randn(2, 512),
        'audio_emb': torch.randn(2, 768),
        'action': torch.randn(2, 4)
    }


@pytest.fixture
def dummy_inputs_single():
    """Single sample dummy inputs"""
    return {
        'image_emb': torch.randn(1, 512),
        'audio_emb': torch.randn(1, 768),
        'action': torch.randn(1, 4)
    }


@pytest.fixture
def dummy_inputs_batch():
    """Batch of dummy inputs"""
    batch_size = 8
    return {
        'image_emb': torch.randn(batch_size, 512),
        'audio_emb': torch.randn(batch_size, 768),
        'action': torch.randn(batch_size, 4)
    }


# ============================================================================
# Training Data Fixtures
# ============================================================================

@pytest.fixture
def echo_mirror_data():
    """Mock EchoMirror training data"""
    num_samples = 20
    return {
        'eeg': torch.randn(num_samples, 8),
        'voice': torch.randn(num_samples, 768),
        'labels': torch.rand(num_samples)
    }


# ============================================================================
# Optimization Fixtures
# ============================================================================

@pytest.fixture
def optimizer(grcm_model):
    """GRCM optimizer"""
    return GRCMOptimizer(grcm_model)


# ============================================================================
# Benchmark Fixtures
# ============================================================================

@pytest.fixture
def benchmark_config():
    """Fast benchmark config for testing"""
    return BenchmarkConfig(
        num_warmup=2,
        num_iterations=5,
        batch_sizes=[1, 2],
        target_latency_ms=50.0
    )


@pytest.fixture
def benchmark(grcm_model, benchmark_config):
    """GRCM benchmark"""
    return GRCMBenchmark(grcm_model, benchmark_config, device="cpu")


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory"""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary model directory"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def has_cuda():
    """Check if CUDA is available"""
    return torch.cuda.is_available()


# ============================================================================
# Marker Helpers
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "stress: Stress tests")
    config.addinivalue_line("markers", "gpu: GPU tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def assert_close():
    """Helper for asserting tensors are close"""
    def _assert_close(a, b, rtol=1e-4, atol=1e-6):
        assert torch.allclose(a, b, rtol=rtol, atol=atol), \
            f"Tensors not close: max diff = {(a - b).abs().max()}"
    return _assert_close


@pytest.fixture
def assert_shape():
    """Helper for asserting tensor shapes"""
    def _assert_shape(tensor, expected_shape):
        assert tensor.shape == expected_shape, \
            f"Shape mismatch: got {tensor.shape}, expected {expected_shape}"
    return _assert_shape


@pytest.fixture
def assert_range():
    """Helper for asserting tensor value range"""
    def _assert_range(tensor, min_val, max_val):
        assert tensor.min() >= min_val and tensor.max() <= max_val, \
            f"Values out of range [{min_val}, {max_val}]: got [{tensor.min()}, {tensor.max()}]"
    return _assert_range
