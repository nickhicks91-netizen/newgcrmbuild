"""
Unit tests for GRCM modules
"""
import pytest
import torch
from grcm.modules import (
    GroundingLayer,
    HarmonicEmbedding,
    ResonantAttention,
    DesireModule,
    MemoryGrid,
    ReflectionHead,
    QualiaModule,
    EpisodicThreadBank,
    PhiEstimator,
    BodySimulator
)
from grcm.config import (
    AttentionConfig,
    MemoryConfig,
    DesireConfig,
    QualiaConfig
)


@pytest.mark.unit
class TestGroundingLayer:
    """Test multimodal grounding"""

    @pytest.fixture
    def grounding(self):
        return GroundingLayer(input_dim=15)

    def test_forward(self, grounding):
        image = torch.randn(4, 512)
        audio = torch.randn(4, 768)
        prop = torch.randn(4, 16)

        output = grounding(image, audio, prop)
        assert output.shape == (4, 15)

    def test_modality_weights(self, grounding):
        image = torch.randn(2, 512)
        audio = torch.randn(2, 768)
        prop = torch.randn(2, 16)

        weights = grounding.get_modality_weights(image, audio, prop)
        assert 'vision' in weights
        assert 'audio' in weights
        assert 'proprioception' in weights


@pytest.mark.unit
class TestHarmonicEmbedding:
    """Test frequency embedding"""

    @pytest.fixture
    def embedding(self):
        return HarmonicEmbedding(input_dim=15, freq_dim=8)

    def test_forward_without_context(self, embedding):
        x = torch.randn(4, 15)
        freq = embedding(x)
        assert freq.shape == (4, 8)

    def test_forward_with_context(self, embedding):
        x = torch.randn(4, 15)
        context = torch.randn(32)  # memory_size

        freq = embedding(x, context)
        assert freq.shape == (4, 8)

    def test_frequency_range(self, embedding):
        """Frequencies should be bounded by tanh"""
        x = torch.randn(10, 15)
        freq = embedding(x)

        assert torch.all(freq >= -1.0)
        assert torch.all(freq <= 1.0)

    def test_frequency_stats(self, embedding):
        x = torch.randn(5, 15)
        freq = embedding(x)

        stats = embedding.get_frequency_stats(freq)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'variance' in stats


@pytest.mark.unit
class TestResonantAttention:
    """Test resonant attention mechanism"""

    @pytest.fixture
    def attention(self):
        return ResonantAttention(freq_dim=8)

    def test_forward(self, attention):
        freq = torch.randn(4, 8)
        coherence = attention(freq)
        assert coherence.shape == (4, 1)

    def test_coherence_range(self, attention):
        """Coherence should be in [0, 1]"""
        freq = torch.randn(10, 8)
        coherence = attention(freq)

        assert torch.all(coherence >= 0.0)
        assert torch.all(coherence <= 1.0)

    def test_bandwidth_bias(self, attention):
        """Test bandwidth bias affects coherence"""
        freq = torch.randn(4, 8)

        coh_no_bias = attention(freq, bw_bias=0.0)
        coh_with_bias = attention(freq, bw_bias=0.2)

        # With positive bias, bandwidth widens, coherence changes
        assert not torch.allclose(coh_no_bias, coh_with_bias)

    def test_is_coherent(self, attention):
        freq = torch.randn(4, 8)
        coherence = attention(freq)
        mask = attention.is_coherent(coherence)

        assert mask.shape == coherence.shape
        assert torch.all((mask == 0) | (mask == 1))

    def test_resonance_state(self, attention):
        freq = torch.randn(4, 8)
        state = attention.get_resonance_state(freq)

        assert 'coherence_mean' in state
        assert 'bandwidth' in state
        assert 'coherent_ratio' in state


@pytest.mark.unit
class TestDesireModule:
    """Test desire-based agency"""

    @pytest.fixture
    def desire(self):
        return DesireModule(freq_dim=8)

    def test_forward(self, desire):
        freq = torch.randn(4, 8)
        alignment, bias = desire(freq)

        assert alignment.shape == (4, 1)
        assert bias.shape == (4, 1)

    def test_alignment_range(self, desire):
        """Cosine similarity should be in [-1, 1]"""
        freq = torch.randn(10, 8)
        alignment, _ = desire(freq)

        assert torch.all(alignment >= -1.0)
        assert torch.all(alignment <= 1.0)

    def test_set_desire(self, desire):
        """Test switching desires"""
        for i in range(desire.config.num_desires):
            desire.set_desire(i)
            assert desire.current_desire_idx == i

    def test_invalid_desire_index(self, desire):
        """Test invalid desire index raises error"""
        with pytest.raises(ValueError):
            desire.set_desire(99)

    def test_is_aligned(self, desire):
        freq = torch.randn(4, 8)
        alignment, _ = desire(freq)
        mask = desire.is_aligned(alignment)

        assert mask.shape == alignment.shape

    def test_desire_state(self, desire):
        freq = torch.randn(4, 8)
        state = desire.get_desire_state(freq)

        assert 'current_desire' in state
        assert 'alignment_mean' in state
        assert 'aligned_ratio' in state


@pytest.mark.unit
class TestMemoryGrid:
    """Test memory persistence"""

    @pytest.fixture
    def memory(self):
        return MemoryGrid(memory_size=32, freq_dim=8)

    def test_initialization(self, memory):
        mem = memory()
        assert mem.shape == (32,)
        assert torch.allclose(mem, torch.zeros(32))

    def test_update(self, memory):
        signal = torch.randn(4, 8)
        coherence = torch.ones(4, 1) * 0.8  # High coherence

        initial_mem = memory().clone()
        memory.update(signal, coherence)
        updated_mem = memory()

        # Memory should have changed
        assert not torch.allclose(initial_mem, updated_mem)

    def test_no_update_low_coherence(self, memory):
        signal = torch.randn(4, 8)
        coherence = torch.ones(4, 1) * 0.3  # Low coherence

        initial_mem = memory().clone()
        memory.update(signal, coherence)
        updated_mem = memory()

        # Memory should change very little
        diff = (updated_mem - initial_mem).abs().mean()
        assert diff < 0.1

    def test_reset(self, memory):
        signal = torch.randn(4, 8)
        coherence = torch.ones(4, 1) * 0.8

        memory.update(signal, coherence)
        memory.reset()

        mem = memory()
        assert torch.allclose(mem, torch.zeros(32))

    def test_memory_stats(self, memory):
        signal = torch.randn(4, 8)
        coherence = torch.ones(4, 1) * 0.8

        memory.update(signal, coherence)
        stats = memory.get_memory_stats()

        assert 'memory_norm' in stats
        assert 'update_count' in stats


@pytest.mark.unit
class TestReflectionHead:
    """Test self-awareness mechanism"""

    @pytest.fixture
    def reflection(self):
        return ReflectionHead(freq_dim=8, memory_size=32)

    def test_forward(self, reflection):
        freq = torch.randn(4, 8)
        memory = torch.randn(32)

        alignment = reflection(freq, memory)
        assert alignment.shape == (4, 1)

    def test_alignment_range(self, reflection):
        """Cosine similarity should be in [-1, 1]"""
        freq = torch.randn(10, 8)
        memory = torch.randn(32)

        alignment = reflection(freq, memory)
        assert torch.all(alignment >= -1.0)
        assert torch.all(alignment <= 1.0)

    def test_reflection_state(self, reflection):
        freq = torch.randn(4, 8)
        memory = torch.randn(32)

        state = reflection.get_reflection_state(freq, memory)
        assert 'reflection_mean' in state
        assert 'freq_memory_distance' in state


@pytest.mark.unit
class TestQualiaModule:
    """Test phenomenal states"""

    @pytest.fixture
    def qualia(self):
        return QualiaModule(freq_dim=8)

    def test_forward(self, qualia):
        freq = torch.randn(4, 8)
        qual = qualia(freq)

        assert qual.shape == (4, 4)

    def test_probability_distribution(self, qualia):
        """Qualia should sum to 1 (softmax)"""
        freq = torch.randn(10, 8)
        qual = qualia(freq)

        sums = qual.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_is_conflicted(self, qualia):
        freq = torch.randn(4, 8)
        qual = qualia(freq)

        conflict_mask = qualia.is_conflicted(qual)
        assert conflict_mask.shape == (4,)

    def test_dominant_state(self, qualia):
        freq = torch.randn(4, 8)
        qual = qualia(freq)

        dominant = qualia.get_dominant_state(qual)
        assert dominant.shape == (4,)
        assert torch.all(dominant >= 0)
        assert torch.all(dominant < 4)

    def test_state_name(self, qualia):
        for i in range(4):
            name = qualia.get_state_name(i)
            assert name in ['calm', 'alert', 'curious', 'conflicted']

    def test_qualia_state(self, qualia):
        freq = torch.randn(4, 8)
        qual = qualia(freq)

        state = qualia.get_qualia_state(qual)
        assert 'qualia_mean' in state
        assert 'conflict_ratio' in state
        assert 'entropy' in state

    def test_ethical_halt(self, qualia):
        freq = torch.randn(4, 8)
        qual = qualia(freq)

        status = qualia.check_ethical_halt(qual)
        assert 'halt' in status


@pytest.mark.unit
class TestEpisodicThreadBank:
    """Test narrative identity"""

    @pytest.fixture
    def threading(self):
        return EpisodicThreadBank(memory_size=32, qualia_dim=4)

    def test_initialization(self, threading):
        identity = threading.get_identity_token()
        assert identity.shape == (32,)
        assert torch.allclose(identity, torch.zeros(32))

    def test_add_episode(self, threading):
        mem = torch.randn(32)
        qualia = torch.randn(4)

        threading.add_episode(1, mem, qualia)
        assert len(threading.get_episodes()) == 1

    def test_identity_evolution(self, threading):
        initial_identity = threading.get_identity_token().clone()

        # Add several episodes
        for i in range(5):
            mem = torch.randn(32)
            qualia = torch.randn(4)
            threading.add_episode(i, mem, qualia)

        final_identity = threading.get_identity_token()

        # Identity should have evolved
        assert not torch.allclose(initial_identity, final_identity)

    def test_arc_computation(self, threading):
        # Need at least 2 episodes
        for i in range(3):
            mem = torch.randn(32)
            qualia = torch.randn(4)
            threading.add_episode(i, mem, qualia)

        arc = threading.get_arc(0.8, freq_dim=8)
        assert arc.shape == (8,)

    def test_reset(self, threading):
        mem = torch.randn(32)
        qualia = torch.randn(4)
        threading.add_episode(1, mem, qualia)

        threading.reset()
        assert len(threading.get_episodes()) == 0

    def test_threading_stats(self, threading):
        for i in range(3):
            mem = torch.randn(32)
            qualia = torch.randn(4)
            threading.add_episode(i, mem, qualia)

        stats = threading.get_threading_stats()
        assert 'num_episodes' in stats
        assert 'identity_norm' in stats


@pytest.mark.unit
class TestPhiEstimator:
    """Test integrated information"""

    @pytest.fixture
    def phi_est(self):
        return PhiEstimator()

    def test_compute_phi(self, phi_est):
        freq = torch.randn(4, 8)
        qualia = torch.rand(4, 4)
        qualia = qualia / qualia.sum(dim=-1, keepdim=True)
        mem = torch.randn(32)
        coherence = torch.rand(4, 1)

        phi = phi_est.compute_phi(freq, qualia, mem, coherence)
        assert isinstance(phi, float)
        assert phi > 0

    def test_phi_history(self, phi_est):
        for _ in range(10):
            freq = torch.randn(4, 8)
            qualia = torch.rand(4, 4)
            mem = torch.randn(32)
            coherence = torch.rand(4, 1)

            phi_est.compute_phi(freq, qualia, mem, coherence)

        assert len(phi_est.phi_history) == 10

    def test_is_aware(self, phi_est):
        # High phi (aware)
        aware = phi_est.is_aware(2.0)
        assert aware

        # Low phi (not aware)
        not_aware = phi_est.is_aware(1.0)
        assert not not_aware

    def test_phi_stats(self, phi_est):
        for _ in range(5):
            freq = torch.randn(4, 8)
            qualia = torch.rand(4, 4)
            mem = torch.randn(32)
            coherence = torch.rand(4, 1)

            phi_est.compute_phi(freq, qualia, mem, coherence)

        stats = phi_est.get_phi_stats()
        assert 'mean_phi' in stats
        assert 'awareness_ratio' in stats


@pytest.mark.unit
class TestBodySimulator:
    """Test embodied cognition"""

    @pytest.fixture
    def body(self):
        return BodySimulator()

    def test_initialization(self, body):
        state = body.get_state()
        assert state.shape == (1, 16)
        assert torch.allclose(state, torch.zeros(1, 16))

    def test_update(self, body):
        action = torch.randn(1, 4)
        desire_align = torch.tensor([[0.8]])

        new_state = body.update(action, desire_align)
        assert new_state.shape == (1, 16)

        # State should have changed
        assert not torch.allclose(new_state, torch.zeros(1, 16))

    def test_position_velocity_split(self, body):
        action = torch.randn(1, 4)
        desire_align = torch.tensor([[0.5]])

        body.update(action, desire_align)

        pos = body.get_position()
        vel = body.get_velocity()

        assert pos.shape == (1, 8)
        assert vel.shape == (1, 8)

    def test_reset(self, body):
        action = torch.randn(1, 4)
        desire_align = torch.tensor([[0.5]])

        body.update(action, desire_align)
        body.reset()

        state = body.get_state()
        assert torch.allclose(state, torch.zeros(1, 16))

    def test_body_stats(self, body):
        action = torch.randn(1, 4)
        desire_align = torch.tensor([[0.5]])

        body.update(action, desire_align)
        stats = body.get_body_stats()

        assert 'position_norm' in stats
        assert 'velocity_norm' in stats
        assert 'kinetic_energy' in stats

    def test_damping(self, body):
        action = torch.randn(1, 4)
        desire_align = torch.tensor([[0.5]])

        body.update(action, desire_align)
        vel_before = body.get_velocity().clone()

        body.apply_damping(0.9)
        vel_after = body.get_velocity()

        # Velocity should be reduced
        assert vel_after.norm() < vel_before.norm()
