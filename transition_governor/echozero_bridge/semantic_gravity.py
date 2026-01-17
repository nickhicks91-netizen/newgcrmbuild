"""
EchoZero Phase 2: Semantic Gravity for Hallucination Correction

Implements "metric engineering" to correct persistent errors (hallucinations)
by treating them as high-mass attractors in semantic space that warp the
manifold of thought.

Key Concepts:
- Hallucination = High-mass object (A >> 1) creating semantic gravity
- Cannot "delete" from hologram (distributed everywhere)
- Solution: Inject counter-mass to balance the metric
- Result: Narrative path pulled away from error toward truth
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .holographic_memory import HolographicMatrix, TorsionalEmbedding


class HallucinationType(Enum):
    """Types of hallucinations detectable via semantic gravity."""
    HIGH_MASS_ATTRACTOR = "high_mass_attractor"  # Persistent error (high amplitude)
    RESONANCE_ANOMALY = "resonance_anomaly"  # Unexpected constructive interference
    METRIC_DISTORTION = "metric_distortion"  # Local geometry warping
    PHASE_INCOHERENCE = "phase_incoherence"  # Temporal inconsistency


@dataclass
class HallucinationSignature:
    """
    Detected hallucination with diagnostic information.
    """
    hallucination_type: HallucinationType
    attractor_indices: List[int]  # Which dimensions are affected
    magnitude: float  # Strength of the distortion
    confidence: float  # Detection confidence (0-1)
    correction_vector: Optional[np.ndarray] = None  # Suggested counter-mass


class SemanticGravityEngine:
    """
    Detects and corrects hallucinations via metric engineering.

    Uses holographic memory to:
    1. Detect high-mass attractors (persistent errors)
    2. Compute semantic gravity field
    3. Inject counter-mass to balance metric
    4. Validate correction effectiveness
    """

    def __init__(
        self,
        hologram: HolographicMatrix,
        attractor_threshold: float = 5.0,
        correction_strength: float = 2.5,
        phase_inversion: bool = True,
        adaptive_threshold: bool = True,
        max_correction_iterations: int = 5
    ):
        """
        Args:
            hologram: HolographicMatrix instance
            attractor_threshold: Multiple of mean magnitude to flag as attractor
            correction_strength: Multiplier for counter-mass (>1 to overpower error)
            phase_inversion: Whether to invert phase for cancellation
            adaptive_threshold: Whether to adjust threshold based on hologram statistics
            max_correction_iterations: Maximum correction passes per auto_correct call
        """
        self.hologram = hologram
        self.attractor_threshold = attractor_threshold
        self.correction_strength = correction_strength
        self.phase_inversion = phase_inversion
        self.adaptive_threshold = adaptive_threshold
        self.max_correction_iterations = max_correction_iterations

        # Tracking
        self.detected_hallucinations: List[HallucinationSignature] = []
        self.correction_history: List[Dict] = []

    def detect_hallucinations(
        self,
        detection_modes: Optional[List[HallucinationType]] = None
    ) -> List[HallucinationSignature]:
        """
        Detect hallucinations via multiple methods.

        Args:
            detection_modes: Which detection methods to use (default: all)

        Returns:
            List of detected hallucination signatures
        """
        if detection_modes is None:
            detection_modes = [
                HallucinationType.HIGH_MASS_ATTRACTOR,
                HallucinationType.METRIC_DISTORTION
            ]

        hallucinations = []

        for mode in detection_modes:
            if mode == HallucinationType.HIGH_MASS_ATTRACTOR:
                hallucinations.extend(self._detect_high_mass_attractors())
            elif mode == HallucinationType.METRIC_DISTORTION:
                hallucinations.extend(self._detect_metric_distortion())

        self.detected_hallucinations.extend(hallucinations)
        return hallucinations

    def _compute_adaptive_threshold(self) -> float:
        """
        Compute adaptive threshold based on hologram statistics.

        Returns higher threshold when hologram has more variance (reduces false positives).
        """
        if not self.adaptive_threshold:
            return self.attractor_threshold

        H_mag = np.abs(self.hologram.H)
        mean_mag = np.mean(H_mag)
        std_mag = np.std(H_mag)

        if mean_mag < 1e-10:
            return self.attractor_threshold

        # Coefficient of variation
        cv = std_mag / (mean_mag + 1e-10)

        # Higher variance → higher threshold (more selective)
        # Scale base threshold by (1 + cv)
        adaptive = self.attractor_threshold * (1.0 + cv)

        return min(adaptive, self.attractor_threshold * 3.0)  # Cap at 3× base

    def _detect_high_mass_attractors(self) -> List[HallucinationSignature]:
        """
        Detect persistent errors via high-mass regions.

        A high-mass attractor indicates repeated enfolding of the same
        (incorrect) pattern, creating semantic gravity that pulls future
        responses toward the error.
        """
        # Use adaptive threshold if enabled
        threshold = self._compute_adaptive_threshold()
        attractors = self.hologram.detect_attractors(threshold=threshold)

        hallucinations = []
        mean_magnitude = np.mean(np.abs(self.hologram.H))

        if mean_magnitude < 1e-10:
            return []  # Empty hologram

        for idx, magnitude in attractors:
            # Confidence based on how much it exceeds mean
            magnitude_ratio = magnitude / (mean_magnitude + 1e-10)
            confidence = min(1.0, magnitude_ratio / threshold)

            # Generate correction vector (full dimensionality)
            # Use stronger correction for higher confidence
            effective_strength = self.correction_strength * (1.0 + confidence)

            correction = np.zeros(self.hologram.dimension, dtype=np.complex128)
            correction[idx] = -self.hologram.H[idx] * effective_strength

            hallucination = HallucinationSignature(
                hallucination_type=HallucinationType.HIGH_MASS_ATTRACTOR,
                attractor_indices=[idx],
                magnitude=magnitude,
                confidence=confidence,
                correction_vector=np.real(correction)
            )

            hallucinations.append(hallucination)

        return hallucinations

    def _detect_metric_distortion(self) -> List[HallucinationSignature]:
        """
        Detect hallucinations via local metric distortion.

        Measures curvature of semantic space. High curvature indicates
        strong gravitational lensing around an error.
        """
        H_mag = np.abs(self.hologram.H)
        mean_mag = np.mean(H_mag)
        std_mag = np.std(H_mag)

        if std_mag < 1e-10:
            return []  # Flat metric, no distortion

        # Compute local curvature (second derivative approximation)
        # High curvature = strong gravity = potential hallucination
        hallucinations = []

        # Sample regions
        window_size = 50
        for i in range(0, len(H_mag) - window_size, window_size):
            window = H_mag[i:i+window_size]
            local_mean = np.mean(window)
            local_std = np.std(window)

            # Curvature indicator: normalized standard deviation
            curvature = local_std / (local_mean + 1e-10)

            if curvature > 2.0:  # High curvature threshold
                # This region is distorted
                affected_indices = list(range(i, i+window_size))
                peak_idx = i + np.argmax(window)
                peak_magnitude = H_mag[peak_idx]

                confidence = min(1.0, curvature / 5.0)

                # Generate correction: smooth out the curvature
                correction = np.zeros(self.hologram.dimension)
                correction[i:i+window_size] = -(window - local_mean) * self.correction_strength

                hallucination = HallucinationSignature(
                    hallucination_type=HallucinationType.METRIC_DISTORTION,
                    attractor_indices=affected_indices,
                    magnitude=peak_magnitude,
                    confidence=confidence,
                    correction_vector=correction
                )

                hallucinations.append(hallucination)

        return hallucinations

    def correct_hallucination(
        self,
        hallucination: HallucinationSignature,
        truth_vector: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Apply counter-mass to correct hallucination.

        Args:
            hallucination: Detected hallucination to correct
            truth_vector: Optional truth embedding (if available)

        Returns:
            Correction statistics
        """
        # Measure pre-correction state
        pre_magnitude = np.linalg.norm(self.hologram.H)
        pre_attractors = len(self.hologram.detect_attractors(self.attractor_threshold))

        # Compute counter-mass vector
        if truth_vector is not None:
            # Use provided truth with high amplitude
            counter_mass = truth_vector
            amplitude = hallucination.magnitude * self.correction_strength
        else:
            # Use computed correction vector
            counter_mass = hallucination.correction_vector
            if counter_mass is None:
                return {'success': False, 'reason': 'no_correction_vector'}
            amplitude = self.correction_strength

        # Inject counter-mass
        self.hologram.inject_counter_mass(
            semantic_vector=counter_mass,
            amplitude=amplitude,
            invert_phase=self.phase_inversion
        )

        # Measure post-correction state
        post_magnitude = np.linalg.norm(self.hologram.H)
        post_attractors = len(self.hologram.detect_attractors(self.attractor_threshold))

        # Check if specific attractor was reduced
        attractor_reduced = False
        if hallucination.attractor_indices:
            idx = hallucination.attractor_indices[0]
            post_attractor_mag = np.abs(self.hologram.H[idx])
            attractor_reduced = post_attractor_mag < hallucination.magnitude * 0.8

        correction_stats = {
            'success': True,
            'hallucination_type': hallucination.hallucination_type.value,
            'pre_magnitude': pre_magnitude,
            'post_magnitude': post_magnitude,
            'magnitude_change': post_magnitude - pre_magnitude,
            'pre_attractors': pre_attractors,
            'post_attractors': post_attractors,
            'attractor_reduced': attractor_reduced,
            'correction_strength_used': amplitude
        }

        self.correction_history.append(correction_stats)
        return correction_stats

    def auto_correct(
        self,
        min_confidence: float = 0.7,
        max_corrections: int = 10,
        iterative: bool = True
    ) -> Dict:
        """
        Automatically detect and correct hallucinations with iterative refinement.

        Args:
            min_confidence: Minimum detection confidence to trigger correction
            max_corrections: Maximum number of corrections to apply per iteration
            iterative: Whether to apply multiple correction passes

        Returns:
            Summary statistics
        """
        total_corrections_applied = 0
        total_successful_corrections = 0
        all_detected = []

        iterations = self.max_correction_iterations if iterative else 1

        for iteration in range(iterations):
            # Detect hallucinations in current state
            hallucinations = self.detect_hallucinations()

            if not hallucinations:
                break  # No more hallucinations detected

            all_detected.extend(hallucinations)

            # Filter by confidence
            high_confidence = [
                h for h in hallucinations
                if h.confidence >= min_confidence
            ]

            if not high_confidence:
                break  # No high-confidence detections

            # Sort by magnitude (correct strongest first)
            high_confidence.sort(key=lambda h: h.magnitude, reverse=True)

            # Apply corrections for this iteration
            corrections_this_iteration = 0
            successful_this_iteration = 0

            for hallucination in high_confidence[:max_corrections]:
                result = self.correct_hallucination(hallucination)
                corrections_this_iteration += 1
                total_corrections_applied += 1

                if result.get('attractor_reduced', False):
                    successful_this_iteration += 1
                    total_successful_corrections += 1

            # Stop if no successful corrections this iteration
            if successful_this_iteration == 0:
                break

        correction_rate = (total_successful_corrections / total_corrections_applied
                          if total_corrections_applied > 0 else 0)

        # Get unique hallucination types
        unique_types = list(set(h.hallucination_type.value for h in all_detected))

        return {
            'total_detected': len(all_detected),
            'unique_detected': len(set((h.attractor_indices[0] if h.attractor_indices else -1)
                                       for h in all_detected)),
            'corrections_applied': total_corrections_applied,
            'successful_corrections': total_successful_corrections,
            'correction_rate': correction_rate,
            'iterations_run': iteration + 1,
            'hallucination_types': unique_types
        }

    def compute_semantic_gravity_field(self) -> np.ndarray:
        """
        Compute the semantic gravity field (curvature of meaning space).

        Returns:
            Vector of "gravitational" force at each dimension
        """
        H_mag = np.abs(self.hologram.H)
        mean_mag = np.mean(H_mag)

        # Gravity proportional to excess mass
        gravity_field = (H_mag - mean_mag) / (mean_mag + 1e-10)

        return gravity_field

    def measure_metric_health(self) -> Dict:
        """
        Measure overall health of semantic space metric.

        Returns:
            Health metrics (0-1, where 1 = perfect health)
        """
        H_mag = np.abs(self.hologram.H)
        mean_mag = np.mean(H_mag)
        std_mag = np.std(H_mag)

        if mean_mag < 1e-10:
            # Empty or near-empty hologram
            return {
                'health_score': 1.0,  # Perfect health (no attractors)
                'curvature': 0.0,
                'flatness': 1.0,
                'num_attractors': 0,
                'mean_gravity': 0.0,
                'max_gravity': 0.0
            }

        # Measure curvature via coefficient of variation
        cv = std_mag / (mean_mag + 1e-10)
        curvature = min(1.0, cv / 2.0)  # Normalize to 0-1

        # Count attractors
        threshold = self._compute_adaptive_threshold()
        num_attractors = len(self.hologram.detect_attractors(threshold))

        # Flatness: inverse of normalized CV (high flatness = good)
        flatness = 1.0 / (1.0 + cv)

        # Attractor penalty: exponential decay
        attractor_penalty = np.exp(-num_attractors / 5.0)  # More aggressive penalty

        # Curvature penalty
        curvature_penalty = 1.0 - curvature

        # Overall health score (0-1, higher is better)
        # Components: flatness (40%), attractor_penalty (40%), curvature_penalty (20%)
        health_score = (
            0.4 * flatness +
            0.4 * attractor_penalty +
            0.2 * curvature_penalty
        )

        # Compute gravity field for additional metrics
        gravity_field = self.compute_semantic_gravity_field()

        return {
            'health_score': np.clip(health_score, 0.0, 1.0),
            'curvature': curvature,
            'flatness': flatness,
            'num_attractors': num_attractors,
            'mean_gravity': np.mean(np.abs(gravity_field)),
            'max_gravity': np.max(np.abs(gravity_field)),
            'attractor_penalty': attractor_penalty
        }

    def get_correction_statistics(self) -> Dict:
        """Get statistics on all corrections performed."""
        if not self.correction_history:
            return {'total_corrections': 0}

        successful = sum(1 for c in self.correction_history if c.get('attractor_reduced', False))

        return {
            'total_corrections': len(self.correction_history),
            'successful_corrections': successful,
            'success_rate': successful / len(self.correction_history),
            'avg_magnitude_change': np.mean([c['magnitude_change'] for c in self.correction_history]),
            'total_attractors_removed': sum(
                c['pre_attractors'] - c['post_attractors']
                for c in self.correction_history
            )
        }


class GovernorWithSemanticGravity:
    """
    Transition Governor + Holographic Memory + Semantic Gravity.

    Phase 2 integration: Governor detects instability, semantic gravity
    corrects the underlying hallucination in memory.
    """

    def __init__(
        self,
        governor,
        hologram: HolographicMatrix,
        enable_auto_correction: bool = True,
        correction_threshold: float = 0.7
    ):
        """
        Args:
            governor: TransitionGovernor instance
            hologram: HolographicMatrix instance
            enable_auto_correction: Whether to auto-correct detected hallucinations
            correction_threshold: Confidence threshold for auto-correction
        """
        self.governor = governor
        self.hologram = hologram
        self.gravity_engine = SemanticGravityEngine(hologram)
        self.enable_auto_correction = enable_auto_correction
        self.correction_threshold = correction_threshold

    def govern_with_correction(
        self,
        state,
        context_embedding: Optional[np.ndarray] = None,
        truth_embedding: Optional[np.ndarray] = None
    ) -> Tuple:
        """
        Govern with hallucination detection and correction.

        Args:
            state: AIState
            context_embedding: Current context
            truth_embedding: Ground truth (if available) for correction

        Returns:
            (governor_output, memory_stats, correction_stats)
        """
        # Standard governance
        output = self.governor.govern(state)

        memory_stats = {}
        correction_stats = {}

        if context_embedding is None:
            return output, memory_stats, correction_stats

        # Store context
        amplitude = 1.0 / (state.entropy + 0.1)
        self.hologram.enfold(context_embedding, amplitude=amplitude)

        # Check metric health
        metric_health = self.gravity_engine.measure_metric_health()
        memory_stats['metric_health'] = metric_health

        # If in brownout or low health, check for hallucinations
        if output.governance_state.value == 'BROWNOUT' or metric_health['health_score'] < 0.5:
            hallucinations = self.gravity_engine.detect_hallucinations()
            memory_stats['detected_hallucinations'] = len(hallucinations)

            if self.enable_auto_correction and hallucinations:
                # Auto-correct high-confidence hallucinations
                high_confidence = [
                    h for h in hallucinations
                    if h.confidence >= self.correction_threshold
                ]

                for hallucination in high_confidence[:3]:  # Limit to 3 per cycle
                    result = self.gravity_engine.correct_hallucination(
                        hallucination,
                        truth_vector=truth_embedding
                    )
                    if not correction_stats:
                        correction_stats = {'corrections': []}
                    correction_stats['corrections'].append(result)

        return output, memory_stats, correction_stats
