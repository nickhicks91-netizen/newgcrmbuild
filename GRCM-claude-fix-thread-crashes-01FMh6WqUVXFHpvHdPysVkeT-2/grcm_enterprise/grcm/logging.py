"""
GRCM MLflow Logging Integration
Experiment tracking, metrics logging, and model versioning
"""
import torch
from typing import Optional, Dict, Any
from pathlib import Path

from .core import ModularGRCM
from .trainer import EchoMirrorTrainer


class MLflowLogger:
    """
    MLflow integration for GRCM experiment tracking

    Logs:
    - Hyperparameters (config)
    - Metrics (phi, coherence, qualia, etc.)
    - Artifacts (models, configs, visualizations)
    - Tags (experiment metadata)
    """

    def __init__(
        self,
        experiment_name: str = "GRCM-Experiments",
        tracking_uri: Optional[str] = None,
        enable: bool = True
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enable = enable
        self.mlflow_available = False
        self.run_id = None

        if self.enable:
            try:
                import mlflow
                self.mlflow = mlflow
                self.mlflow_available = True

                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)

                mlflow.set_experiment(experiment_name)
            except ImportError:
                print("[MLflow] Warning: mlflow not installed. Logging disabled.")
                print("[MLflow] Install with: pip install mlflow")
                self.enable = False

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """Start MLflow run"""
        if not self.enable or not self.mlflow_available:
            return

        self.mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = self.mlflow.active_run().info.run_id
        print(f"[MLflow] Started run: {self.run_id}")

    def end_run(self):
        """End MLflow run"""
        if not self.enable or not self.mlflow_available:
            return

        self.mlflow.end_run()
        print(f"[MLflow] Ended run: {self.run_id}")

    def log_config(self, config):
        """Log GRCM configuration"""
        if not self.enable or not self.mlflow_available:
            return

        config_dict = config.to_dict() if hasattr(config, 'to_dict') else config

        # Flatten nested config
        flat_params = self._flatten_dict(config_dict, parent_key='config')

        self.mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics at specific step"""
        if not self.enable or not self.mlflow_available:
            return

        self.mlflow.log_metrics(metrics, step=step)

    def log_model_outputs(self, outputs: Dict[str, Any], step: int):
        """Log GRCM forward pass outputs"""
        if not self.enable or not self.mlflow_available:
            return

        metrics = {
            'phi': outputs['phi'],
            'coherence_mean': outputs['coherence'].mean().item(),
            'coherence_std': outputs['coherence'].std().item(),
            'desire_align_mean': outputs['desire_align'].mean().item(),
            'reflection_mean': outputs['reflection'].mean().item(),
        }

        # Qualia distribution
        qualia_mean = outputs['qualia'].mean(0)
        for i, label in enumerate(['calm', 'alert', 'curious', 'conflicted']):
            metrics[f'qualia_{label}'] = qualia_mean[i].item()

        self.mlflow.log_metrics(metrics, step=step)

    def log_training_metrics(
        self,
        epoch: int,
        loss: float,
        phi: float,
        alignment_error: Optional[float] = None
    ):
        """Log training metrics"""
        if not self.enable or not self.mlflow_available:
            return

        metrics = {
            'train_loss': loss,
            'train_phi': phi
        }

        if alignment_error is not None:
            metrics['train_alignment_error'] = alignment_error

        self.mlflow.log_metrics(metrics, step=epoch)

    def log_benchmark_results(self, benchmark_results: Dict[str, Any]):
        """Log benchmark results"""
        if not self.enable or not self.mlflow_available:
            return

        # Flatten and log
        flat_metrics = self._flatten_dict(benchmark_results, parent_key='benchmark')

        # Filter to numeric values only
        numeric_metrics = {
            k: v for k, v in flat_metrics.items()
            if isinstance(v, (int, float))
        }

        self.mlflow.log_metrics(numeric_metrics)

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact file"""
        if not self.enable or not self.mlflow_available:
            return

        self.mlflow.log_artifact(artifact_path, artifact_name)

    def log_model(
        self,
        model: ModularGRCM,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """Log PyTorch model"""
        if not self.enable or not self.mlflow_available:
            return

        # Save model state dict
        model_path = Path("temp_model.pt")
        torch.save(model.state_dict(), model_path)

        self.mlflow.log_artifact(str(model_path), artifact_path)

        # Clean up
        model_path.unlink()

        # Optionally register model
        if registered_model_name:
            self.mlflow.pytorch.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )

    def log_config_artifact(self, config, filename: str = "config.yaml"):
        """Log configuration as YAML artifact"""
        if not self.enable or not self.mlflow_available:
            return

        temp_path = Path(filename)
        config.to_yaml(str(temp_path))

        self.mlflow.log_artifact(str(temp_path))

        temp_path.unlink()

    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                # Skip lists/tuples for now
                continue
            else:
                items.append((new_key, v))

        return dict(items)

    def __enter__(self):
        """Context manager entry"""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.end_run()


class GRCMExperiment:
    """
    High-level experiment wrapper with MLflow logging

    Simplifies running and tracking GRCM experiments
    """

    def __init__(
        self,
        model: ModularGRCM,
        experiment_name: str = "GRCM-Experiment",
        enable_logging: bool = True
    ):
        self.model = model
        self.logger = MLflowLogger(experiment_name, enable=enable_logging)

    def run_forward_experiment(
        self,
        num_steps: int = 100,
        batch_size: int = 4,
        log_interval: int = 10
    ):
        """Run forward pass experiment with logging"""
        with self.logger:
            # Log configuration
            self.logger.log_config(self.model.config)

            # Run forward passes
            for step in range(num_steps):
                # Generate inputs
                image_emb = torch.randn(batch_size, 512)
                audio_emb = torch.randn(batch_size, 768)
                action = torch.randn(batch_size, 4)

                # Forward pass
                outputs = self.model(image_emb, audio_emb, action)

                # Log at intervals
                if step % log_interval == 0:
                    self.logger.log_model_outputs(outputs, step)

            print(f"[Experiment] Completed {num_steps} forward passes")

    def run_training_experiment(
        self,
        eeg_data: torch.Tensor,
        voice_data: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int = 10,
        run_name: Optional[str] = None
    ):
        """Run training experiment with logging"""
        with self.logger:
            self.logger.start_run(run_name=run_name or "echo_mirror_training")

            # Log configuration
            self.logger.log_config(self.model.config)

            # Create trainer
            trainer = EchoMirrorTrainer(self.model)

            # Training loop with logging
            for epoch in range(num_epochs):
                metrics = trainer.train_epoch(eeg_data, voice_data, labels)

                # Log metrics
                self.logger.log_training_metrics(
                    epoch=epoch,
                    loss=metrics['loss'],
                    phi=metrics['phi'],
                    alignment_error=metrics.get('alignment_error')
                )

            # Log final model
            self.logger.log_model(self.model, registered_model_name="GRCM")

            # Get and log summary
            summary = trainer.get_training_summary()
            self.logger.log_metrics({
                'final_loss': summary['final_loss'],
                'final_phi': summary['final_phi'],
                'loss_improvement': summary['loss_improvement']
            })

            print(f"[Experiment] Training complete")


def quick_mlflow_experiment(
    config_path: str = "config/grcm_default.yaml",
    num_steps: int = 50,
    experiment_name: str = "GRCM-Quick-Test"
):
    """
    Quick MLflow experiment for testing

    Args:
        config_path: Path to GRCM config
        num_steps: Number of forward passes
        experiment_name: MLflow experiment name
    """
    from .core import ModularGRCM
    from .config import load_config

    # Load model
    config = load_config(config_path)
    model = ModularGRCM(config)

    # Run experiment
    experiment = GRCMExperiment(model, experiment_name)
    experiment.run_forward_experiment(num_steps=num_steps)

    print("\n[MLflow] View results with:")
    print("  mlflow ui")
    print("  # Then open http://localhost:5000")
