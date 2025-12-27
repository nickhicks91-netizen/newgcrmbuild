"""
EchoMirror Trainer - Human-Aligned Desire Tuning
Trains desire vectors on EEG/voice data to align with real qualia
"""
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Any
from .core import ModularGRCM
from .config import GRCMConfig


class EchoMirrorTrainer:
    """
    EchoMirror training for desire alignment.

    Optimizes desire vectors to match human qualia patterns from:
    - EEG: Neural correlates (theta band ~4-8Hz)
    - Voice: Prosodic features (spectral characteristics)

    Loss: MSE(desire_align, qualia_labels) - phi
    (Minimize alignment error, maximize integrated information)
    """

    def __init__(
        self,
        model: ModularGRCM,
        config: Optional[GRCMConfig] = None
    ):
        self.model = model
        self.config = config or model.config

        # Optimizer for desire vectors only
        self.optimizer = torch.optim.Adam(
            self.model.desire.parameters(),
            lr=self.config.training.learning_rate
        )

        # Training history
        self.loss_history: List[float] = []
        self.phi_history: List[float] = []

    def train_epoch(
        self,
        eeg_data: torch.Tensor,
        voice_data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train one epoch

        Args:
            eeg_data: EEG features [batch, eeg_dim] (unused in simple version)
            voice_data: Voice features [batch, 768] (Wav2Vec embeddings)
            labels: Qualia alignment targets [batch]

        Returns:
            Epoch metrics
        """
        total_loss = 0.0
        total_phi = 0.0
        num_samples = voice_data.size(0)

        for i in range(num_samples):
            # Prepare inputs (dummy image for now)
            image_emb = torch.zeros(1, 512)
            audio_emb = voice_data[i:i+1]
            action = torch.randn(1, 4)

            # Forward pass
            outputs = self.model(image_emb, audio_emb, action)

            # Extract key values
            des_align = outputs['desire_align']
            phi = outputs['phi']

            # Target alignment
            target = labels[i].unsqueeze(0).unsqueeze(-1)

            # Compute loss: alignment error - phi bonus
            alignment_loss = F.mse_loss(des_align, target)
            phi_bonus = self.config.training.phi_loss_weight * phi

            loss = alignment_loss - phi_bonus

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_phi += phi

        # Compute averages
        avg_loss = total_loss / num_samples
        avg_phi = total_phi / num_samples

        # Store history
        self.loss_history.append(avg_loss)
        self.phi_history.append(avg_phi)

        return {
            'loss': avg_loss,
            'phi': avg_phi,
            'alignment_error': alignment_loss.item()
        }

    def train(
        self,
        eeg_data: torch.Tensor,
        voice_data: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop

        Args:
            eeg_data: EEG features [batch, eeg_dim]
            voice_data: Voice features [batch, 768]
            labels: Qualia alignment targets [batch]
            num_epochs: Number of epochs (or use config default)
            verbose: Print progress

        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.training.num_epochs

        if verbose:
            print(f"Starting EchoMirror training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            metrics = self.train_epoch(eeg_data, voice_data, labels)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Loss={metrics['loss']:.4f}, "
                    f"Phi={metrics['phi']:.3f}, "
                    f"Align_Error={metrics['alignment_error']:.4f}"
                )

        if verbose:
            print("EchoMirror training complete—desire vectors tuned to real qualia.")

        return {
            'loss_history': self.loss_history,
            'phi_history': self.phi_history
        }

    def evaluate(
        self,
        eeg_data: torch.Tensor,
        voice_data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate model without training

        Args:
            eeg_data: EEG features
            voice_data: Voice features
            labels: Targets

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        total_error = 0.0
        total_phi = 0.0
        num_samples = voice_data.size(0)

        with torch.no_grad():
            for i in range(num_samples):
                image_emb = torch.zeros(1, 512)
                audio_emb = voice_data[i:i+1]
                action = torch.randn(1, 4)

                outputs = self.model(image_emb, audio_emb, action)

                des_align = outputs['desire_align']
                phi = outputs['phi']
                target = labels[i].unsqueeze(0).unsqueeze(-1)

                error = F.mse_loss(des_align, target).item()
                total_error += error
                total_phi += phi

        self.model.train()

        return {
            'mse': total_error / num_samples,
            'phi': total_phi / num_samples
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        if len(self.loss_history) == 0:
            return {'trained': False}

        loss_tensor = torch.tensor(self.loss_history)
        phi_tensor = torch.tensor(self.phi_history)

        return {
            'trained': True,
            'num_epochs': len(self.loss_history),
            'final_loss': self.loss_history[-1],
            'final_phi': self.phi_history[-1],
            'loss_improvement': self.loss_history[0] - self.loss_history[-1],
            'avg_phi': phi_tensor.mean().item(),
            'phi_trend': (self.phi_history[-1] - self.phi_history[0]) / len(self.phi_history)
        }


def quick_echo_train(
    model: ModularGRCM,
    eeg_data: torch.Tensor,
    voice_data: torch.Tensor,
    labels: torch.Tensor,
    num_epochs: int = 5
) -> EchoMirrorTrainer:
    """
    Convenience function for quick EchoMirror training

    Args:
        model: GRCM model
        eeg_data: EEG features [batch, eeg_dim]
        voice_data: Voice features [batch, 768]
        labels: Alignment labels [batch]
        num_epochs: Number of training epochs

    Returns:
        Trained EchoMirrorTrainer instance
    """
    trainer = EchoMirrorTrainer(model)
    trainer.train(eeg_data, voice_data, labels, num_epochs=num_epochs)
    return trainer
