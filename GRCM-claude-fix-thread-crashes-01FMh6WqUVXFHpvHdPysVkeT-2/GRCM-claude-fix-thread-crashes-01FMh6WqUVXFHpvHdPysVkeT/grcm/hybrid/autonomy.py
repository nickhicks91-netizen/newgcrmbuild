"""
Autonomy loop with continual learning.

Background process that:
1. Continuously runs forward passes
2. Updates coupling via EchoMirror
3. Logs phi and system metrics
4. Adapts to new data streams
"""

import torch
import time
import threading
from typing import Optional, Dict, Callable
from queue import Queue
import logging

from .forward import EchoGRCMHybrid
from ..train import EchoMirrorTrainer, MultimodalDatastream


logger = logging.getLogger(__name__)


class AutonomyLoop:
    """
    Autonomous learning loop for EchoGRCM.

    Runs in background thread, continuously:
    - Generating/receiving data
    - Running forward passes
    - Updating coupling via EchoMirror
    - Logging system state
    """

    def __init__(
        self,
        model: EchoGRCMHybrid,
        trainer: EchoMirrorTrainer,
        datastream: MultimodalDatastream,
        update_interval: float = 1.0,
        log_interval: int = 10,
        callback: Optional[Callable] = None,
    ):
        """
        Initialize autonomy loop.

        Args:
            model: EchoGRCM model
            trainer: EchoMirror trainer
            datastream: Data source
            update_interval: Time between updates (seconds)
            log_interval: Steps between logging
            callback: Optional callback(step, outputs, stats)
        """
        self.model = model
        self.trainer = trainer
        self.datastream = datastream
        self.update_interval = update_interval
        self.log_interval = log_interval
        self.callback = callback

        self.running = False
        self.thread = None
        self.step_count = 0

        self.metrics = {
            "phi": [],
            "coherence": [],
            "coupling_norm": [],
        }

    def start(self):
        """Start autonomy loop in background thread."""
        if self.running:
            logger.warning("Autonomy loop already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        logger.info("Autonomy loop started")

    def stop(self):
        """Stop autonomy loop."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5.0)

        logger.info(f"Autonomy loop stopped after {self.step_count} steps")

    def _loop(self):
        """Main loop (runs in background thread)."""
        logger.info("Autonomy loop running...")

        while self.running:
            try:
                # Generate data
                batch = self.datastream.generate_batch()

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**batch)

                # Hebbian update
                K_new, stats = self.trainer.batch_update(
                    K=self.model.echozero.K,
                    psi_batch=outputs["psi"],
                    coherence_batch=outputs["coherence"],
                )

                # Update coupling IN-PLACE
                self.model.echozero.K.copy_(K_new)

                # Log metrics
                self.metrics["phi"].append(outputs["phi"].mean().item())
                self.metrics["coherence"].append(outputs["coherence"].mean().item())
                self.metrics["coupling_norm"].append(stats["coupling_norm"])

                # Logging
                if self.step_count % self.log_interval == 0:
                    logger.info(
                        f"Step {self.step_count}: "
                        f"Φ={outputs['phi'].mean().item():.4f}, "
                        f"Coh={outputs['coherence'].mean().item():.4f}"
                    )

                # Callback
                if self.callback is not None:
                    self.callback(self.step_count, outputs, stats)

                self.step_count += 1

                # Sleep
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in autonomy loop: {e}")
                time.sleep(1.0)

    def get_metrics(self) -> Dict[str, any]:
        """Get current metrics."""
        return {
            "step_count": self.step_count,
            "metrics": self.metrics,
            "running": self.running,
        }


def run_autonomy(
    model: EchoGRCMHybrid,
    duration: float = 60.0,
    update_interval: float = 0.1,
) -> Dict[str, any]:
    """
    Run autonomy loop for specified duration.

    Args:
        model: Model to run
        duration: Duration in seconds
        update_interval: Update interval

    Returns:
        results: Metrics and statistics
    """
    trainer = EchoMirrorTrainer()
    datastream = MultimodalDatastream()

    loop = AutonomyLoop(
        model=model,
        trainer=trainer,
        datastream=datastream,
        update_interval=update_interval,
    )

    loop.start()

    try:
        time.sleep(duration)
    finally:
        loop.stop()

    return loop.get_metrics()
