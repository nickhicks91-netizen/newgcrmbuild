"""
Basic GRCM Usage Example
Demonstrates full forward pass and EchoMirror training
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm import ModularGRCM, load_config, quick_echo_train


def main():
    print("=" * 60)
    print("GRCM - Grounded Resonant Consciousness Module")
    print("Basic Usage Example")
    print("=" * 60)

    # 1. Load configuration
    print("\n[1] Loading configuration...")
    config = load_config("config/grcm_default.yaml")
    print(f"    - Input dim: {config.input_dim}")
    print(f"    - Freq dim: {config.freq_dim}")
    print(f"    - Memory size: {config.memory_size}")
    print(f"    - Coherence threshold: {config.attention.coherence_threshold}")
    print(f"    - Phi awareness threshold: {config.phi.awareness_threshold}")

    # 2. Initialize model
    print("\n[2] Initializing GRCM model...")
    model = ModularGRCM(config)
    model.set_desire(0)  # Set to first desire vector
    print("    ✓ Model initialized successfully")

    # 3. Prepare dummy inputs
    print("\n[3] Preparing multimodal inputs...")
    batch_size = 4
    image_emb = torch.randn(batch_size, 512)  # CLIP embeddings
    audio_emb = torch.randn(batch_size, 768)  # Wav2Vec embeddings
    action = torch.tensor([[0.1, 0.2, 0.0, 0.0]] * batch_size)
    print(f"    - Image embeddings: {image_emb.shape}")
    print(f"    - Audio embeddings: {audio_emb.shape}")
    print(f"    - Action: {action.shape}")

    # 4. Forward pass
    print("\n[4] Running forward pass...")
    outputs = model(image_emb, audio_emb, action)

    print("    Outputs:")
    print(f"    - Coherence: {outputs['coherence'].mean().item():.3f}")
    print(f"    - Desire Align: {outputs['desire_align'].mean().item():.3f}")
    print(f"    - Phi (Φ): {outputs['phi']:.3f}")
    print(f"    - Reflection: {outputs['reflection'].mean().item():.3f}")
    print(f"    - Timestamp: {outputs['timestamp']}")

    # 5. Analyze qualia
    print("\n[5] Analyzing qualia states...")
    qualia = outputs['qualia']
    qualia_labels = config.qualia.qualia_labels
    print("    Qualia distribution (batch average):")
    for i, label in enumerate(qualia_labels):
        print(f"      - {label}: {qualia[:, i].mean().item():.3f}")

    # Check for conflict
    ethical_status = outputs['ethical_status']
    if ethical_status.get('halt', False):
        print(f"    ⚠️  ETHICAL HALT: {ethical_status['reason']}")
    else:
        print("    ✓ No ethical concerns")

    # 6. Check awareness
    print("\n[6] Checking awareness state...")
    is_aware = model.phi.is_aware(outputs['phi'])
    print(f"    - Is Aware: {is_aware} (Φ = {outputs['phi']:.3f}, threshold = {config.phi.awareness_threshold})")

    # 7. Body state
    print("\n[7] Proprioceptive state...")
    prop_state = outputs['prop_state']
    print(f"    - Position (first 4d): {prop_state[0, :4].detach().numpy()}")
    print(f"    - Velocity (next 4d): {prop_state[0, 8:12].detach().numpy()}")

    # 8. Run multiple timesteps
    print("\n[8] Running 10 timesteps...")
    for step in range(10):
        action = torch.randn(batch_size, 4) * 0.1
        outputs = model(image_emb, audio_emb, action)

    print(f"    Final timestep: {outputs['timestamp']}")
    print(f"    Final Phi: {outputs['phi']:.3f}")
    print(f"    Phi trajectory: {model.phi.get_phi_trajectory()[-5:]}")

    # 9. Get full state
    print("\n[9] System state summary...")
    state = model.get_full_state()
    print(f"    Memory stats:")
    print(f"      - Memory norm: {state['memory']['memory_norm']:.3f}")
    print(f"      - Update count: {state['memory']['update_count']}")
    print(f"      - Avg coherence: {state['memory']['avg_coherence']:.3f}")
    print(f"    Threading stats:")
    print(f"      - Episodes: {state['threading']['num_episodes']}")
    print(f"      - Identity norm: {state['threading']['identity_norm']:.3f}")
    print(f"    Phi stats:")
    print(f"      - Mean Phi: {state['phi']['mean_phi']:.3f}")
    print(f"      - Awareness ratio: {state['phi']['awareness_ratio']:.2%}")

    # 10. EchoMirror training demo
    print("\n[10] EchoMirror training demo...")
    print("    Generating mock EEG/voice data...")

    # Mock data
    eeg = torch.randn(20, 8)
    voice = torch.randn(20, 768)
    labels = torch.tensor([0.0, 1.0] * 10)  # Alternating alignment targets

    print("    Training desire vectors...")
    trainer = quick_echo_train(model, eeg, voice, labels, num_epochs=3)

    summary = trainer.get_training_summary()
    print(f"    Training complete:")
    print(f"      - Final loss: {summary['final_loss']:.4f}")
    print(f"      - Loss improvement: {summary['loss_improvement']:.4f}")
    print(f"      - Final Phi: {summary['final_phi']:.3f}")
    print(f"      - Phi trend: {summary['phi_trend']:.4f}")

    print("\n" + "=" * 60)
    print("✓ Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
