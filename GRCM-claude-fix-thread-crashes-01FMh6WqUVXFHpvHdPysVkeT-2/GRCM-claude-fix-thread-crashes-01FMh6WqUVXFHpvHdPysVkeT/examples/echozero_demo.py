"""
EchoZero + GRCM demonstration script.

Shows basic usage of the hybrid system.
"""

import torch
from grcm.hybrid import EchoGRCMHybrid
from grcm.train import (
    EchoMirrorTrainer,
    MultimodalDatastream,
    training_loop,
)


def main():
    """Run EchoZero demonstration."""
    print("=" * 60)
    print("  EchoZero + GRCM Hybrid System Demonstration")
    print("=" * 60)

    # 1. Initialize system
    print("\n[1/5] Initializing EchoGRCM system...")
    model = EchoGRCMHybrid(
        n_nodes=64,
        grounded_dim=15,
        memory_dim=128,
        device="cpu",
    )
    print(f"✓ System initialized with {model.n_nodes} nodes")

    # 2. Generate test data
    print("\n[2/5] Generating test data...")
    datastream = MultimodalDatastream(batch_size=1)
    batch = datastream.generate_batch()
    print("✓ Data generated: image, audio, action vectors")

    # 3. Forward pass
    print("\n[3/5] Running forward pass...")
    with torch.no_grad():
        outputs = model(**batch)

    print(f"✓ Forward pass complete")
    print(f"  Φ (Integrated Information): {outputs['phi'][0].item():.4f}")
    print(f"  Mean Coherence: {outputs['coherence'][0].mean().item():.4f}")
    print(f"  Qualia: {outputs['qualia'][0].tolist()}")
    print(f"  Dominant: {['Calm', 'Alert', 'Curious', 'Conflicted'][outputs['qualia'][0].argmax()]}")

    # 4. EchoMirror training
    print("\n[4/5] Running EchoMirror Hebbian training...")
    trainer = EchoMirrorTrainer(learning_rate=0.001)

    stats = training_loop(
        model=model,
        datastream=datastream,
        trainer=trainer,
        n_steps=100,
        log_interval=25,
    )

    print(f"✓ Training complete")
    print(f"  Final Φ: {stats['final_phi']:.4f}")
    print(f"  Training speed: {stats['steps_per_sec']:.2f} steps/sec")

    # 5. System info
    print("\n[5/5] System information:")
    info = model.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("  Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
