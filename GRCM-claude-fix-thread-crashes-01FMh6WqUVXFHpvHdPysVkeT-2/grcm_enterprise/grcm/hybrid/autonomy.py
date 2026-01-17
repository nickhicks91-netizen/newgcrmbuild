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
from typing import Optional, Dict, Callable, Any
from queue import Queue
import logging
import copy

from .forward import EchoGRCMHybrid
from ..train import EchoMirrorTrainer, MultimodalDatastream


logger = logging.getLogger(__name__)


class ThreadSafeMetrics:
    """Thread-safe container for autonomy loop metrics."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._metrics = {
            "phi": [],
            "coherence": [],
            "coupling_norm": [],
        }
        self._step_count = 0
    
    def append(self, key: str, value: float):
        """Thread-safe append to metrics list."""
        with self._lock:
            if key in self._metrics:
                self._metrics[key].append(value)
    
    def increment_step(self) -> int:
        """Increment step count and return new value."""
        with self._lock:
            self._step_count += 1
            return self._step_count
    
    def get_step_count(self) -> int:
        """Get current step count."""
        with self._lock:
            return self._step_count
    
    def get_metrics_snapshot(self) -> Dict:
        """Get a thread-safe copy of all metrics."""
        with self._lock:
            return {
                "step_count": self._step_count,
                "metrics": copy.deepcopy(self._metrics),
            }


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
        
        self._model_lock = threading.RLock()
        self._thread_safe_metrics = ThreadSafeMetrics()

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

        step_count = self._thread_safe_metrics.get_step_count()
        logger.info(f"Autonomy loop stopped after {step_count} steps")

    def _loop(self):
        """Main loop (runs in background thread)."""
        logger.info("Autonomy loop running...")

        while self.running:
            try:
                # Generate data
                batch = self.datastream.generate_batch()

                # Forward pass and Hebbian update with model lock
                with self._model_lock:
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

                # Log metrics (thread-safe)
                self._thread_safe_metrics.append("phi", outputs["phi"].mean().item())
                self._thread_safe_metrics.append("coherence", outputs["coherence"].mean().item())
                self._thread_safe_metrics.append("coupling_norm", stats["coupling_norm"])

                step_count = self._thread_safe_metrics.increment_step()

                # Logging
                if step_count % self.log_interval == 0:
                    logger.info(
                        f"Step {step_count}: "
                        f"Φ={outputs['phi'].mean().item():.4f}, "
                        f"Coh={outputs['coherence'].mean().item():.4f}"
                    )

                # Callback
                if self.callback is not None:
                    self.callback(step_count, outputs, stats)

                # Sleep
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in autonomy loop: {e}")
                time.sleep(1.0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics (thread-safe)."""
        snapshot = self._thread_safe_metrics.get_metrics_snapshot()
        return {
            "step_count": snapshot["step_count"],
            "metrics": snapshot["metrics"],
            "running": self.running,
        }
    
    @property
    def step_count(self) -> int:
        """Get current step count (backward compatibility)."""
        return self._thread_safe_metrics.get_step_count()
    
    def update_metric(self, key: str, value: float) -> None:
        """Thread-safe update of a single metric value."""
        self._thread_safe_metrics.append(key, value)
    
    def get_metrics_copy(self) -> Dict:
        """Get a thread-safe copy of metrics for reading."""
        return self._thread_safe_metrics.get_metrics_snapshot()["metrics"]


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
