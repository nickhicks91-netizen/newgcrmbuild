"""
Unit tests for core GRCM module
"""
import pytest
import torch
from grcm import ModularGRCM, GRCMConfig


@pytest.mark.unit
class TestModularGRCM:
    """Test ModularGRCM core functionality"""

    def test_initialization(self, default_config):
        """Test model initialization"""
        model = ModularGRCM(default_config)
        assert model.config == default_config
        assert model.t == 0

    def test_forward_pass(self, grcm_model, dummy_inputs_small):
        """Test forward pass"""
        outputs = grcm_model(**dummy_inputs_small)

        # Check all expected keys present
        expected_keys = [
            'output', 'coherence', 'memory', 'reflection',
            'qualia', 'identity_token', 'desire_align',
            'phi', 'prop_state', 'ethical_status', 'timestamp'
        ]
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"

    def test_output_shapes(self, grcm_model, dummy_inputs_small):
        """Test output tensor shapes"""
        batch_size = dummy_inputs_small['image_emb'].size(0)
        outputs = grcm_model(**dummy_inputs_small)

        assert outputs['output'].shape == (batch_size, grcm_model.config.memory_size)
        assert outputs['coherence'].shape == (batch_size, 1)
        assert outputs['reflection'].shape == (batch_size, 1)
        assert outputs['qualia'].shape == (batch_size, 4)
        assert outputs['desire_align'].shape == (batch_size, 1)

    def test_phi_computation(self, grcm_model, dummy_inputs_small):
        """Test Phi computation"""
        outputs = grcm_model(**dummy_inputs_small)
        phi = outputs['phi']

        assert isinstance(phi, float)
        assert phi > 0  # Phi should be positive

    def test_coherence_range(self, grcm_model, dummy_inputs_small):
        """Test coherence is in valid range [0, 1]"""
        outputs = grcm_model(**dummy_inputs_small)
        coherence = outputs['coherence']

        assert torch.all(coherence >= 0.0)
        assert torch.all(coherence <= 1.0)

    def test_qualia_distribution(self, grcm_model, dummy_inputs_small):
        """Test qualia forms valid probability distribution"""
        outputs = grcm_model(**dummy_inputs_small)
        qualia = outputs['qualia']

        # Check sums to 1 (softmax output)
        sums = qualia.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

        # Check all positive
        assert torch.all(qualia >= 0.0)

    def test_desire_setting(self, grcm_model):
        """Test desire index setting"""
        # Set to different desires
        for i in range(grcm_model.config.desire.num_desires):
            grcm_model.set_desire(i)
            assert grcm_model.desire.current_desire_idx == i

    def test_reset_functionality(self, grcm_model, dummy_inputs_small):
        """Test model reset"""
        # Run forward pass
        _ = grcm_model(**dummy_inputs_small)
        assert grcm_model.t > 0

        # Reset
        grcm_model.reset()
        assert grcm_model.t == 0
        assert len(grcm_model.phi.phi_history) == 0

    def test_multiple_forward_passes(self, grcm_model, dummy_inputs_small):
        """Test multiple consecutive forward passes"""
        phi_values = []

        for i in range(5):
            outputs = grcm_model(**dummy_inputs_small)
            phi_values.append(outputs['phi'])
            assert outputs['timestamp'] == i + 1

        # Check phi history length
        assert len(grcm_model.phi.phi_history) == 5

    def test_ethical_halt_detection(self, grcm_model):
        """Test ethical halt mechanism"""
        # This is harder to test reliably, but we can check the structure
        outputs = grcm_model(
            torch.randn(1, 512),
            torch.randn(1, 768),
            torch.randn(1, 4)
        )

        ethical_status = outputs['ethical_status']
        assert isinstance(ethical_status, dict)
        assert 'halt' in ethical_status or len(ethical_status) == 0

    def test_get_full_state(self, grcm_model):
        """Test full state retrieval"""
        state = grcm_model.get_full_state()

        expected_keys = ['timestamp', 'memory', 'threading', 'phi', 'body', 'config']
        for key in expected_keys:
            assert key in state

    def test_from_config_file(self, temp_config_file):
        """Test loading from config file"""
        model = ModularGRCM.from_config_file(temp_config_file)
        assert model.config.input_dim == 10
        assert model.config.freq_dim == 4

    def test_device_compatibility(self, grcm_model):
        """Test model works on CPU"""
        model = grcm_model.to("cpu")
        outputs = model(
            torch.randn(1, 512),
            torch.randn(1, 768),
            torch.randn(1, 4)
        )
        assert outputs['output'].device.type == "cpu"

    @pytest.mark.gpu
    def test_gpu_compatibility(self, grcm_model, has_cuda):
        """Test model works on GPU if available"""
        if not has_cuda:
            pytest.skip("CUDA not available")

        model = grcm_model.to("cuda")
        outputs = model(
            torch.randn(1, 512).cuda(),
            torch.randn(1, 768).cuda(),
            torch.randn(1, 4).cuda()
        )
        assert outputs['output'].device.type == "cuda"

    def test_batch_size_flexibility(self, grcm_model):
        """Test different batch sizes"""
        for batch_size in [1, 2, 4, 8]:
            outputs = grcm_model(
                torch.randn(batch_size, 512),
                torch.randn(batch_size, 768),
                torch.randn(batch_size, 4)
            )
            assert outputs['output'].size(0) == batch_size

    def test_no_grad_mode(self, grcm_model, dummy_inputs_small):
        """Test model works in no_grad mode"""
        with torch.no_grad():
            outputs = grcm_model(**dummy_inputs_small)
            assert not outputs['output'].requires_grad

    def test_memory_persistence(self, grcm_model, dummy_inputs_small):
        """Test memory persists across forward passes"""
        # First forward pass
        outputs1 = grcm_model(**dummy_inputs_small)
        mem1 = outputs1['memory'].clone()

        # Second forward pass
        outputs2 = grcm_model(**dummy_inputs_small)
        mem2 = outputs2['memory']

        # Memory should have changed
        assert not torch.allclose(mem1, mem2)

    def test_identity_evolution(self, grcm_model, dummy_inputs_small):
        """Test identity token evolves over time"""
        initial_identity = grcm_model.threading.get_identity_token().clone()

        # Run several forward passes with high coherence
        for _ in range(10):
            _ = grcm_model(**dummy_inputs_small)

        final_identity = grcm_model.threading.get_identity_token()

        # Identity should have evolved
        assert not torch.allclose(initial_identity, final_identity)


@pytest.mark.unit
class TestGRCMConfig:
    """Test GRCM configuration"""

    def test_default_config(self):
        """Test default configuration"""
        config = GRCMConfig()
        assert config.input_dim == 15
        assert config.freq_dim == 8
        assert config.memory_size == 32

    def test_custom_config(self):
        """Test custom configuration"""
        config = GRCMConfig(
            input_dim=20,
            freq_dim=12,
            memory_size=64
        )
        assert config.input_dim == 20
        assert config.freq_dim == 12
        assert config.memory_size == 64

    def test_config_to_dict(self, default_config):
        """Test config to dict conversion"""
        config_dict = default_config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'input_dim' in config_dict
        assert 'attention' in config_dict

    def test_config_from_dict(self):
        """Test config from dict"""
        config_dict = {
            'input_dim': 10,
            'freq_dim': 6,
            'attention': {'base_bandwidth': 0.3}
        }
        config = GRCMConfig.from_dict(config_dict)
        assert config.input_dim == 10
        assert config.attention.base_bandwidth == 0.3

    def test_config_yaml_roundtrip(self, tmp_path):
        """Test config YAML save/load"""
        config = GRCMConfig(input_dim=25)
        yaml_path = tmp_path / "test_config.yaml"

        # Save
        config.to_yaml(str(yaml_path))

        # Load
        loaded_config = GRCMConfig.from_yaml(str(yaml_path))
        assert loaded_config.input_dim == 25
