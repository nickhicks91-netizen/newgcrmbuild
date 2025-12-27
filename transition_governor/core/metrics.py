"""
Entropy and confidence metric computation.

This module provides adapters for extracting stability signals from various sources.
All computations are deterministic and free of interpretation.
"""

from typing import Protocol, Optional
import numpy as np
from abc import ABC, abstractmethod


class EntropyProvider(Protocol):
    """
    Protocol for entropy signal sources.

    The governor does NOT care where entropy comes from.
    Implementations may use logits, logprobs, or proxies.
    """

    def get_entropy(self) -> float:
        """
        Compute current entropy.

        Returns:
            Non-negative float. Higher = more uncertainty.
        """
        ...

    def get_entropy_dot(self) -> float:
        """
        Compute rate of change of entropy.

        Returns:
            Float (may be negative). Large magnitude = rapid transition.
        """
        ...


class DirectEntropyAdapter(EntropyProvider):
    """
    Entropy computation from token probability distributions.

    Use this for HuggingFace models or any source with logits access.

    Invariants:
    - Entropy is computed as H = -Σ p_i log(p_i)
    - No smoothing or interpretation
    - Deterministic from probabilities
    """

    def __init__(self, seed: int = 42):
        """
        Initialize adapter.

        Args:
            seed: Random seed for deterministic behavior (if needed for sampling)
        """
        self.seed = seed
        self.history: list[float] = []
        self.max_history = 100  # Limit memory usage

    def get_entropy(self, logits: Optional[np.ndarray] = None, probs: Optional[np.ndarray] = None) -> float:
        """
        Compute entropy from logits or probabilities.

        Args:
            logits: Raw model output logits (will be softmaxed)
            probs: Pre-computed probabilities (must sum to ~1.0)

        Returns:
            Entropy in nats (natural log)
        """
        if probs is None:
            if logits is None:
                raise ValueError("Must provide either logits or probs")
            # Softmax with numerical stability
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits)

        # Validate probability distribution
        assert np.abs(np.sum(probs) - 1.0) < 1e-6, "Probabilities must sum to 1.0"
        assert np.all(probs >= 0.0), "Probabilities must be non-negative"

        # Compute entropy: H = -Σ p_i log(p_i)
        # Avoid log(0) by filtering out zero probabilities
        nonzero_probs = probs[probs > 0]
        entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))

        # Store in history
        self.history.append(entropy)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return float(entropy)

    def get_entropy_dot(self) -> float:
        """
        Compute rate of change from history.

        Returns:
            Difference between current and previous entropy.
            Zero if insufficient history.
        """
        if len(self.history) < 2:
            return 0.0
        return self.history[-1] - self.history[-2]


class ProxyEntropyAdapter(EntropyProvider):
    """
    Entropy estimation from API proxies (logprobs, token variance, etc).

    Use this for OpenAI or streaming APIs without logits access.

    Proxy signals:
    - logprobs (lower = higher confidence = lower entropy)
    - token variance (higher = more uncertainty = higher entropy)
    - response length (abnormally short/long may indicate instability)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.history: list[float] = []
        self.max_history = 100

    def get_entropy(self, logprob: Optional[float] = None, token_variance: Optional[float] = None) -> float:
        """
        Estimate entropy from available proxies.

        Args:
            logprob: Log probability of most likely token (negative)
            token_variance: Variance in token probabilities

        Returns:
            Estimated entropy (non-negative)
        """
        if logprob is not None:
            # Convert logprob to entropy proxy
            # Lower logprob (more negative) = higher uncertainty
            entropy = -logprob  # Now positive
        elif token_variance is not None:
            # Higher variance = higher entropy
            entropy = token_variance
        else:
            raise ValueError("Must provide at least one proxy signal")

        # Store in history
        self.history.append(entropy)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return float(entropy)

    def get_entropy_dot(self) -> float:
        """Compute rate of change from history."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1] - self.history[-2]


class ConfidenceEstimator:
    """
    Confidence computation from entropy.

    Confidence is the inverse of entropy (normalized).
    This is NOT a semantic measure of "correctness" - it's a stability signal.

    Invariants:
    - confidence ∈ [0, 1]
    - confidence = exp(-k * entropy) for some constant k
    - Lower entropy → higher confidence
    """

    def __init__(self, decay_constant: float = 1.0):
        """
        Initialize estimator.

        Args:
            decay_constant: Controls entropy-to-confidence mapping sensitivity
        """
        self.decay_constant = decay_constant

    def estimate_confidence(self, entropy: float) -> float:
        """
        Convert entropy to confidence.

        Args:
            entropy: Current entropy value (non-negative)

        Returns:
            Confidence in [0, 1]
        """
        assert entropy >= 0.0, "Entropy must be non-negative"
        confidence = np.exp(-self.decay_constant * entropy)
        return float(np.clip(confidence, 0.0, 1.0))

    def estimate_confidence_dot(self, entropy_dot: float, current_confidence: float) -> float:
        """
        Compute rate of change of confidence from entropy rate.

        Args:
            entropy_dot: Rate of change of entropy
            current_confidence: Current confidence value

        Returns:
            Rate of change of confidence
        """
        # d/dt[exp(-k*H)] = -k * exp(-k*H) * dH/dt = -k * confidence * entropy_dot
        confidence_dot = -self.decay_constant * current_confidence * entropy_dot
        return float(confidence_dot)


class ToolStateProvider(Protocol):
    """
    Protocol for tool execution state monitoring.

    The governor monitors tool state for stability signals only.
    It does NOT interpret tool semantics or results.
    """

    def current_state(self) -> 'ToolState':
        """
        Get current tool execution state.

        Returns:
            ToolState enum value
        """
        ...


# Import ToolState from state module
from .state import ToolState


class SimpleToolStateMonitor(ToolStateProvider):
    """
    Basic tool state monitor.

    Tracks tool lifecycle: inactive → pending → active → (success|error) → inactive

    Failure modes monitored:
    - Timeouts (pending too long)
    - Errors (exceptions, failures)
    - Stuck states (active too long)
    """

    def __init__(self, timeout_threshold: float = 30.0):
        """
        Initialize monitor.

        Args:
            timeout_threshold: Seconds before pending/active is considered stuck
        """
        self._state = ToolState.INACTIVE
        self.timeout_threshold = timeout_threshold
        self._state_start_time: Optional[float] = None

    def current_state(self) -> ToolState:
        """Get current state."""
        return self._state

    def transition_to(self, new_state: ToolState, timestamp: float) -> None:
        """
        Transition to new state.

        Args:
            new_state: Target state
            timestamp: Current time (for timeout detection)
        """
        self._state = new_state
        self._state_start_time = timestamp

    def check_timeout(self, current_time: float) -> bool:
        """
        Check if current state has exceeded timeout.

        Args:
            current_time: Current timestamp

        Returns:
            True if timed out
        """
        if self._state_start_time is None:
            return False

        if self._state in (ToolState.PENDING, ToolState.ACTIVE):
            elapsed = current_time - self._state_start_time
            return elapsed > self.timeout_threshold

        return False

    def mark_error(self) -> None:
        """Mark tool as errored."""
        self._state = ToolState.ERROR
