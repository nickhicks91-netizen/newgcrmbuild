"""
Training loop for EchoMirror.

Orchestrates data streaming, forward pass, and Hebbian updates.
"""

import torch
from typing import Dict, Optional, Callable
import time
from tqdm import tqdm

from ..hybrid import EchoGRCMHybrid
from .echo_mirror import EchoMirrorTrainer
from .datastream import MultimodalDatastream


def training_loop(
    model: EchoGRCMHybrid,
    datastream: MultimodalDatastream,
    trainer: EchoMirrorTrainer,
    n_steps: int = 1000,
    log_interval: int = 100,
    callback: Optional[Callable] = None,
    device: str = "cpu",
) -> Dict[str, any]:
    """
    Main training loop for EchoMirror.

    NO backpropagation. Pure Hebbian updates.

    Args:
        model: EchoGRCM hybrid model
        datastream: Multimodal data generator
        trainer: EchoMirror Hebbian trainer
        n_steps: Number of training steps
        log_interval: Logging frequency
        callback: Optional callback(step, outputs, stats)
        device: Computation device

    Returns:
        training_stats: Dictionary of training statistics
    """
    model.to(device)
    model.eval()  # No gradient computation needed

    # Statistics tracking
    stats_history = {
        "phi": [],
        "coherence": [],
        "coupling_norm": [],
        "delta_norm": [],
        "qualia_distribution": [],
    }

    print(f"Starting EchoMirror training for {n_steps} steps...")
    print(f"Model: {model.n_nodes} nodes, device: {device}")

    start_time = time.time()

    with torch.no_grad():  # NO gradients!
        for step in tqdm(range(n_steps), desc="Training"):
            # Generate batch
            batch = datastream.generate_batch()

            # Forward pass
            outputs = model(
                image_emb=batch["image_emb"],
                audio_emb=batch["audio_emb"],
                action=batch["action"],
            )

            # Hebbian coupling update
            K_new, update_stats = trainer.batch_update(
                K=model.echozero.K,
                psi_batch=outputs["psi"],
                coherence_batch=outputs["coherence"],
            )

            # Update coupling matrix IN-PLACE (no optimizer!)
            model.echozero.K.copy_(K_new)

            # Log statistics
            stats_history["phi"].append(outputs["phi"].mean().item())
            stats_history["coherence"].append(outputs["coherence"].mean().item())
            stats_history["coupling_norm"].append(update_stats["coupling_norm"])
            stats_history["delta_norm"].append(update_stats["delta_norm"])

            qualia_dist = outputs["qualia"].mean(dim=0).cpu().numpy().tolist()
            stats_history["qualia_distribution"].append(qualia_dist)

            # Logging
            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed

                print(f"\nStep {step + 1}/{n_steps}")
                print(f"  Φ: {outputs['phi'].mean().item():.4f}")
                print(f"  Coherence: {outputs['coherence'].mean().item():.4f}")
                print(f"  Coupling norm: {update_stats['coupling_norm']:.4f}")
                print(f"  Delta norm: {update_stats['delta_norm']:.6f}")
                print(f"  Qualia: {qualia_dist}")
                print(f"  Speed: {steps_per_sec:.2f} steps/sec")

            # Callback
            if callback is not None:
                callback(step, outputs, update_stats)

    # Final statistics
    total_time = time.time() - start_time

    final_stats = {
        "n_steps": n_steps,
        "total_time": total_time,
        "steps_per_sec": n_steps / total_time,
        "history": stats_history,
        "trainer_stats": trainer.get_statistics(),
        "final_phi": stats_history["phi"][-1] if stats_history["phi"] else 0,
        "final_coherence": stats_history["coherence"][-1] if stats_history["coherence"] else 0,
    }

    print(f"\nTraining complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average speed: {n_steps / total_time:.2f} steps/sec")
    print(f"Final Φ: {final_stats['final_phi']:.4f}")

    return final_stats


def eval_mode(
    model: EchoGRCMHybrid,
    datastream: MultimodalDatastream,
    n_samples: int = 100,
    device: str = "cpu",
) -> Dict[str, any]:
    """
    Evaluate model without training.

    Args:
        model: Model to evaluate
        datastream: Data source
        n_samples: Number of samples
        device: Device

    Returns:
        eval_stats: Evaluation statistics
    """
    model.to(device)
    model.eval()

    stats = {
        "phi": [],
        "coherence": [],
        "qualia": [],
    }

    with torch.no_grad():
        for _ in range(n_samples):
            batch = datastream.generate_batch()

            outputs = model(
                image_emb=batch["image_emb"],
                audio_emb=batch["audio_emb"],
                action=batch["action"],
            )

            stats["phi"].append(outputs["phi"].mean().item())
            stats["coherence"].append(outputs["coherence"].mean().item())
            stats["qualia"].append(outputs["qualia"].mean(dim=0).cpu().numpy())

    eval_stats = {
        "mean_phi": torch.tensor(stats["phi"]).mean().item(),
        "std_phi": torch.tensor(stats["phi"]).std().item(),
        "mean_coherence": torch.tensor(stats["coherence"]).mean().item(),
        "std_coherence": torch.tensor(stats["coherence"]).std().item(),
        "mean_qualia": torch.tensor(stats["qualia"]).mean(dim=0).numpy().tolist(),
    }

    return eval_stats
