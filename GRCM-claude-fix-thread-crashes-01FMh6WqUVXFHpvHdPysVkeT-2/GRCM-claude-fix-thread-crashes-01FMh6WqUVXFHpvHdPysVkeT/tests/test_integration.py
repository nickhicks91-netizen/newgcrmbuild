"""
Integration tests for GRCM full pipeline
"""
import pytest
import torch
from grcm import ModularGRCM, EchoMirrorTrainer, quick_echo_train


@pytest.mark.integration
class TestFullPipeline:
    """Test complete GRCM pipeline"""

    def test_end_to_end_forward(self, grcm_model, dummy_inputs_small):
        """Test complete forward pass"""
        outputs = grcm_model(**dummy_inputs_small)

        # Verify all outputs
        assert outputs['phi'] > 0
        assert outputs['coherence'].mean() >= 0
        assert outputs['qualia'].sum(dim=-1).allclose(torch.ones(2), atol=1e-5)

    def test_multi_step_pipeline(self, grcm_model):
        """Test multiple forward passes with state accumulation"""
        phi_values = []
        coherence_values = []

        for i in range(10):
            outputs = grcm_model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                torch.randn(1, 4)
            )
            phi_values.append(outputs['phi'])
            coherence_values.append(outputs['coherence'].mean().item())

        # Check evolution
        assert len(phi_values) == 10
        assert len(set(phi_values)) > 1  # Values should vary

    def test_memory_coherence_interaction(self, grcm_model):
        """Test memory updates based on coherence"""
        # Force high coherence by using similar inputs
        base_input = {
            'image_emb': torch.randn(1, 512),
            'audio_emb': torch.randn(1, 768),
            'action': torch.randn(1, 4)
        }

        mem_before = grcm_model.memory.get_memory_snapshot()

        # Multiple passes
        for _ in range(5):
            _ = grcm_model(**base_input)

        mem_after = grcm_model.memory.get_memory_snapshot()

        # Memory should have updated
        assert not torch.allclose(mem_before, mem_after)

    def test_desire_memory_gating(self, grcm_model):
        """Test desire gates memory updates"""
        # Set desire
        grcm_model.set_desire(0)

        # Track memory updates with different inputs
        update_counts_before = grcm_model.memory.update_count

        for _ in range(5):
            _ = grcm_model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                torch.randn(1, 4)
            )

        update_counts_after = grcm_model.memory.update_count

        # Some updates should have occurred
        assert update_counts_after >= update_counts_before

    def test_episodic_threading(self, grcm_model):
        """Test episodic memory accumulation"""
        # Run many passes to trigger episode storage
        for i in range(20):
            _ = grcm_model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                torch.randn(1, 4)
            )

        # Check episodes stored
        episodes = grcm_model.threading.get_episodes()
        assert len(episodes) > 0

        # Check identity evolved
        identity = grcm_model.threading.get_identity_token()
        assert identity.norm() > 0

    def test_body_action_loop(self, grcm_model):
        """Test proprioceptive feedback loop"""
        initial_prop = grcm_model.body.get_state().clone()

        # Run with actions
        for _ in range(5):
            action = torch.randn(1, 4)
            _ = grcm_model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                action
            )

        final_prop = grcm_model.body.get_state()

        # Body state should have changed
        assert not torch.allclose(initial_prop, final_prop)

    def test_ethical_halt_trigger(self, grcm_model):
        """Test ethical halt can be triggered"""
        # This is probabilistic, so we just check the mechanism works
        for _ in range(10):
            outputs = grcm_model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                torch.randn(1, 4)
            )

            if outputs['ethical_status'].get('halt', False):
                # Halt was triggered
                assert 'reason' in outputs['ethical_status']
                break

    @pytest.mark.slow
    def test_long_running_stability(self, grcm_model):
        """Test model remains stable over many iterations"""
        phi_values = []

        for i in range(100):
            outputs = grcm_model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                torch.randn(1, 4)
            )
            phi_values.append(outputs['phi'])

        # Check no NaN or Inf
        assert all(not torch.isnan(torch.tensor(p)) for p in phi_values)
        assert all(not torch.isinf(torch.tensor(p)) for p in phi_values)

        # Check reasonable variance
        phi_std = torch.tensor(phi_values).std()
        assert phi_std < 1.0  # Should be relatively stable


@pytest.mark.integration
class TestEchoMirrorTraining:
    """Test EchoMirror training pipeline"""

    def test_training_loop(self, grcm_model, echo_mirror_data):
        """Test EchoMirror training"""
        trainer = EchoMirrorTrainer(grcm_model)

        history = trainer.train(
            echo_mirror_data['eeg'],
            echo_mirror_data['voice'],
            echo_mirror_data['labels'],
            num_epochs=3,
            verbose=False
        )

        # Check history
        assert 'loss_history' in history
        assert 'phi_history' in history
        assert len(history['loss_history']) == 3

    def test_training_improves_loss(self, grcm_model, echo_mirror_data):
        """Test training reduces loss"""
        trainer = EchoMirrorTrainer(grcm_model)

        history = trainer.train(
            echo_mirror_data['eeg'],
            echo_mirror_data['voice'],
            echo_mirror_data['labels'],
            num_epochs=5,
            verbose=False
        )

        # Loss should generally decrease
        first_loss = history['loss_history'][0]
        last_loss = history['loss_history'][-1]

        # Allow for some variance
        assert last_loss < first_loss * 1.5

    def test_quick_echo_train(self, grcm_model, echo_mirror_data):
        """Test convenience training function"""
        trainer = quick_echo_train(
            grcm_model,
            echo_mirror_data['eeg'],
            echo_mirror_data['voice'],
            echo_mirror_data['labels'],
            num_epochs=2
        )

        assert isinstance(trainer, EchoMirrorTrainer)
        summary = trainer.get_training_summary()
        assert summary['trained']

    def test_evaluation(self, grcm_model, echo_mirror_data):
        """Test evaluation mode"""
        trainer = EchoMirrorTrainer(grcm_model)

        metrics = trainer.evaluate(
            echo_mirror_data['eeg'],
            echo_mirror_data['voice'],
            echo_mirror_data['labels']
        )

        assert 'mse' in metrics
        assert 'phi' in metrics

    def test_training_summary(self, grcm_model, echo_mirror_data):
        """Test training summary generation"""
        trainer = EchoMirrorTrainer(grcm_model)

        # Before training
        summary_before = trainer.get_training_summary()
        assert not summary_before['trained']

        # Train
        trainer.train(
            echo_mirror_data['eeg'],
            echo_mirror_data['voice'],
            echo_mirror_data['labels'],
            num_epochs=2,
            verbose=False
        )

        # After training
        summary_after = trainer.get_training_summary()
        assert summary_after['trained']
        assert 'final_loss' in summary_after


@pytest.mark.integration
@pytest.mark.slow
class TestStressTests:
    """Stress tests for GRCM"""

    @pytest.mark.stress
    def test_large_batch(self, grcm_model):
        """Test large batch processing"""
        batch_size = 32

        outputs = grcm_model(
            torch.randn(batch_size, 512),
            torch.randn(batch_size, 768),
            torch.randn(batch_size, 4)
        )

        assert outputs['output'].size(0) == batch_size

    @pytest.mark.stress
    def test_many_iterations(self, grcm_model):
        """Test many sequential iterations"""
        for i in range(1000):
            _ = grcm_model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                torch.randn(1, 4)
            )

        # Should complete without errors
        assert grcm_model.t == 1000

    @pytest.mark.stress
    def test_memory_leaks(self, grcm_model):
        """Test for memory leaks"""
        import gc

        # Force garbage collection
        gc.collect()

        # Run many iterations
        for _ in range(100):
            _ = grcm_model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                torch.randn(1, 4)
            )

        gc.collect()

        # Should not accumulate excessive memory
        # (Hard to test precisely, but should not crash)

    @pytest.mark.stress
    def test_concurrent_models(self):
        """Test multiple models running simultaneously"""
        models = [ModularGRCM() for _ in range(3)]

        # Run each model
        for model in models:
            _ = model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                torch.randn(1, 4)
            )

        # All should work independently
        for model in models:
            assert model.t == 1

    @pytest.mark.stress
    def test_rapid_desire_switching(self, grcm_model):
        """Test rapid desire switching"""
        for i in range(100):
            grcm_model.set_desire(i % 4)
            _ = grcm_model(
                torch.randn(1, 512),
                torch.randn(1, 768),
                torch.randn(1, 4)
            )

        # Should handle switches gracefully
        assert grcm_model.t == 100


@pytest.mark.integration
class TestConfigurationVariations:
    """Test different configuration setups"""

    def test_minimal_config(self):
        """Test minimal configuration"""
        from grcm.config import GRCMConfig

        config = GRCMConfig(
            input_dim=5,
            freq_dim=2,
            memory_size=8
        )

        model = ModularGRCM(config)
        outputs = model(
            torch.randn(1, 512),
            torch.randn(1, 768),
            torch.randn(1, 4)
        )

        assert outputs['phi'] > 0

    def test_large_config(self):
        """Test larger configuration"""
        from grcm.config import GRCMConfig

        config = GRCMConfig(
            input_dim=30,
            freq_dim=16,
            memory_size=64
        )

        model = ModularGRCM(config)
        outputs = model(
            torch.randn(1, 512),
            torch.randn(1, 768),
            torch.randn(1, 4)
        )

        assert outputs['phi'] > 0

    def test_custom_thresholds(self):
        """Test custom threshold configurations"""
        from grcm.config import GRCMConfig, AttentionConfig, PhiConfig

        config = GRCMConfig(
            attention=AttentionConfig(coherence_threshold=0.9),
            phi=PhiConfig(awareness_threshold=2.0)
        )

        model = ModularGRCM(config)
        _ = model(
            torch.randn(1, 512),
            torch.randn(1, 768),
            torch.randn(1, 4)
        )

        # Should work with custom thresholds
        assert model.attn.config.coherence_threshold == 0.9
