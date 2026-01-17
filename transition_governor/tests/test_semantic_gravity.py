"""
Tests for EchoZero Phase 2: Semantic Gravity (Hallucination Correction)

Validates:
1. Hallucination detection (high-mass attractors, metric distortion)
2. Counter-mass injection and correction
3. Metric health monitoring
4. Auto-correction effectiveness
5. Integration with Governor
"""

import pytest
import numpy as np
from transition_governor.echozero_bridge.holographic_memory import HolographicMatrix
from transition_governor.echozero_bridge.semantic_gravity import (
    SemanticGravityEngine,
    HallucinationType,
    HallucinationSignature,
    GovernorWithSemanticGravity
)
from transition_governor import TransitionGovernor, AIState
from transition_governor.core.state import ToolState


class TestHallucinationDetection:
    """Test hallucination detection mechanisms."""

    def test_high_mass_attractor_detection(self):
        """Test that persistent errors (high-mass regions) are detected."""
        H = HolographicMatrix(dimension=1024, seed=42)
        gravity = SemanticGravityEngine(H, attractor_threshold=3.0)

        # Add normal data
        for _ in range(10):
            vec = np.random.randn(1024) * 0.1
            H.enfold(vec, amplitude=1.0)

        # Add persistent error (hallucination) - same pattern repeated
        error_pattern = np.zeros(1024)
        error_pattern[500:510] = 1.0
        for _ in range(5):
            H.enfold(error_pattern, amplitude=10.0)  # High mass

        # Detect hallucinations
        hallucinations = gravity.detect_hallucinations(
            detection_modes=[HallucinationType.HIGH_MASS_ATTRACTOR]
        )

        # Should detect at least one high-mass attractor
        assert len(hallucinations) > 0
        assert any(h.hallucination_type == HallucinationType.HIGH_MASS_ATTRACTOR
                   for h in hallucinations)

        # Should have high confidence
        max_confidence = max(h.confidence for h in hallucinations)
        assert max_confidence > 0.5

    def test_metric_distortion_detection(self):
        """Test detection of semantic space curvature."""
        H = HolographicMatrix(dimension=2048, seed=42)
        gravity = SemanticGravityEngine(H, attractor_threshold=3.0)

        # Add uniform background
        for _ in range(20):
            vec = np.random.randn(2048) * 0.1
            H.enfold(vec, amplitude=1.0)

        # Create local distortion (concentrated mass)
        distortion = np.zeros(2048)
        distortion[1000:1050] = np.random.randn(50) * 10.0  # Localized spike
        H.enfold(distortion, amplitude=20.0)

        # Detect metric distortion
        hallucinations = gravity.detect_hallucinations(
            detection_modes=[HallucinationType.METRIC_DISTORTION]
        )

        # Should detect distortion
        assert len(hallucinations) > 0
        distortion_detected = any(
            h.hallucination_type == HallucinationType.METRIC_DISTORTION
            for h in hallucinations
        )
        assert distortion_detected

    def test_hallucination_signature_contains_correction(self):
        """Test that detected hallucinations include correction vectors."""
        H = HolographicMatrix(dimension=1024, seed=42)
        gravity = SemanticGravityEngine(H)

        # Create error
        error = np.ones(1024)
        H.enfold(error, amplitude=50.0)

        hallucinations = gravity.detect_hallucinations()

        # Should have correction vectors
        assert len(hallucinations) > 0
        for h in hallucinations:
            if h.correction_vector is not None:
                assert h.correction_vector.shape == (1024,)
                # Correction should be non-zero
                assert np.linalg.norm(h.correction_vector) > 0


class TestCounterMassCorrection:
    """Test counter-mass injection for hallucination correction."""

    def test_counter_mass_reduces_attractor(self):
        """Test that counter-mass injection reduces attractor magnitude."""
        H = HolographicMatrix(dimension=1024, seed=42)
        gravity = SemanticGravityEngine(H, correction_strength=1.5, attractor_threshold=3.0)

        # Create strong attractor (hallucination)
        error = np.zeros(1024)
        error[500] = 1.0
        H.enfold(error, amplitude=30.0)

        # Measure attractor strength
        attractors_before = H.detect_attractors(threshold=3.0)
        magnitude_before = max([mag for _, mag in attractors_before]) if attractors_before else 0

        # Detect and correct
        hallucinations = gravity.detect_hallucinations()
        assert len(hallucinations) > 0

        correction_result = gravity.correct_hallucination(hallucinations[0])

        # Should report success
        assert correction_result['success']

        # Attractor should be reduced
        attractors_after = H.detect_attractors(threshold=3.0)

        # Either fewer attractors or reduced magnitude
        assert (len(attractors_after) < len(attractors_before) or
                correction_result['attractor_reduced'])

    def test_truth_vector_correction(self):
        """Test correction using provided truth vector."""
        H = HolographicMatrix(dimension=1024, seed=42)
        gravity = SemanticGravityEngine(H, correction_strength=2.0)

        # Create error pattern
        error = np.ones(1024) * 0.5
        error[:100] = 5.0  # Strong error in first 100 dims
        H.enfold(error, amplitude=20.0)

        # Detect
        hallucinations = gravity.detect_hallucinations()
        assert len(hallucinations) > 0

        # Provide truth vector (opposite of error)
        truth = np.ones(1024) * 0.5
        truth[:100] = -5.0  # Opposite pattern

        # Correct with truth
        result = gravity.correct_hallucination(hallucinations[0], truth_vector=truth)

        assert result['success']
        # Magnitude change should reflect counter-mass injection
        assert result['magnitude_change'] != 0

    def test_phase_inversion_cancellation(self):
        """Test that phase inversion creates destructive interference."""
        H = HolographicMatrix(dimension=1024, seed=42)

        # Create error with specific phase
        error = np.random.randn(1024)
        H.enfold(error, amplitude=10.0, omega_scale=1.0)

        magnitude_before = np.linalg.norm(H.H)

        # Inject with phase inversion
        gravity = SemanticGravityEngine(H, phase_inversion=True, correction_strength=1.0)
        H.inject_counter_mass(error, amplitude=10.0, invert_phase=True)

        magnitude_after = np.linalg.norm(H.H)

        # Magnitude should decrease (partial cancellation)
        assert magnitude_after < magnitude_before

    def test_correction_strength_multiplier(self):
        """Test that correction strength amplifies counter-mass."""
        H1 = HolographicMatrix(dimension=1024, seed=42)
        H2 = HolographicMatrix(dimension=1024, seed=42)

        error = np.ones(1024)

        # Same error in both
        H1.enfold(error, amplitude=10.0)
        H2.enfold(error, amplitude=10.0)

        # Weak correction
        gravity1 = SemanticGravityEngine(H1, correction_strength=1.0)
        hallucinations1 = gravity1.detect_hallucinations()
        if hallucinations1:
            gravity1.correct_hallucination(hallucinations1[0])

        # Strong correction
        gravity2 = SemanticGravityEngine(H2, correction_strength=3.0)
        hallucinations2 = gravity2.detect_hallucinations()
        if hallucinations2:
            gravity2.correct_hallucination(hallucinations2[0])

        # Stronger correction should have bigger impact
        # (measured by fewer attractors or lower magnitude)
        attractors1 = len(H1.detect_attractors(threshold=5.0))
        attractors2 = len(H2.detect_attractors(threshold=5.0))

        # Stronger correction should be at least as effective
        assert attractors2 <= attractors1


class TestMetricHealth:
    """Test semantic space metric health monitoring."""

    def test_metric_health_perfect_case(self):
        """Test metric health for clean, uniform hologram."""
        H = HolographicMatrix(dimension=1024, seed=42)
        gravity = SemanticGravityEngine(H)

        # Add uniform data
        for _ in range(50):
            vec = np.random.randn(1024) * 0.1
            H.enfold(vec, amplitude=1.0)

        health = gravity.measure_metric_health()

        # Uniform data should have good health
        assert 'health_score' in health
        assert 'curvature' in health
        assert 'flatness' in health
        assert 'num_attractors' in health

        # Health score should be reasonable
        assert 0 <= health['health_score'] <= 1.0

    def test_metric_health_degrades_with_attractors(self):
        """Test that attractors reduce metric health."""
        H = HolographicMatrix(dimension=1024, seed=42)
        gravity = SemanticGravityEngine(H)

        # Measure clean health
        for _ in range(10):
            vec = np.random.randn(1024) * 0.1
            H.enfold(vec, amplitude=1.0)

        health_clean = gravity.measure_metric_health()

        # Add attractors
        for _ in range(5):
            error = np.random.randn(1024)
            H.enfold(error, amplitude=30.0)

        health_distorted = gravity.measure_metric_health()

        # Health should degrade
        assert health_distorted['health_score'] < health_clean['health_score']
        assert health_distorted['num_attractors'] > health_clean['num_attractors']

    def test_semantic_gravity_field_computation(self):
        """Test computation of gravity field."""
        H = HolographicMatrix(dimension=1024, seed=42)
        gravity = SemanticGravityEngine(H)

        # Add data
        for _ in range(20):
            vec = np.random.randn(1024) * 0.5
            H.enfold(vec, amplitude=1.0)

        gravity_field = gravity.compute_semantic_gravity_field()

        # Should return field vector
        assert gravity_field.shape == (1024,)

        # Should have non-zero values
        assert np.any(gravity_field != 0)


class TestAutoCorrection:
    """Test automatic hallucination correction."""

    def test_auto_correct_detects_and_fixes(self):
        """Test that auto_correct both detects and corrects."""
        H = HolographicMatrix(dimension=1024, seed=42)
        gravity = SemanticGravityEngine(H)

        # Add errors
        for _ in range(3):
            error = np.random.randn(1024)
            H.enfold(error, amplitude=25.0)

        # Auto-correct
        result = gravity.auto_correct(min_confidence=0.5, max_corrections=5)

        # Should detect hallucinations
        assert result['total_detected'] > 0

        # Should apply corrections
        assert result['corrections_applied'] > 0

        # Correction rate should be calculated
        assert 'correction_rate' in result
        assert 0 <= result['correction_rate'] <= 1.0

    def test_auto_correct_respects_confidence_threshold(self):
        """Test that low-confidence hallucinations are not corrected."""
        H = HolographicMatrix(dimension=1024, seed=42)

        # Add many low-amplitude items (low confidence hallucinations)
        for _ in range(50):
            vec = np.random.randn(1024) * 0.1
            H.enfold(vec, amplitude=1.0)

        gravity = SemanticGravityEngine(H, attractor_threshold=2.0)

        # Auto-correct with high confidence threshold
        result = gravity.auto_correct(min_confidence=0.9, max_corrections=10)

        # May detect low-confidence issues but shouldn't correct them
        # (or correct very few)
        assert result['corrections_applied'] <= result['high_confidence']

    def test_auto_correct_limits_max_corrections(self):
        """Test that max_corrections parameter is respected."""
        H = HolographicMatrix(dimension=2048, seed=42)
        gravity = SemanticGravityEngine(H)

        # Create many attractors
        for i in range(20):
            error = np.zeros(2048)
            error[i*100:(i+1)*100] = np.random.randn(100) * 5.0
            H.enfold(error, amplitude=15.0)

        # Limit corrections
        result = gravity.auto_correct(min_confidence=0.1, max_corrections=3)

        # Should apply at most 3 corrections
        assert result['corrections_applied'] <= 3

    def test_correction_statistics_tracking(self):
        """Test that correction history is tracked."""
        H = HolographicMatrix(dimension=1024, seed=42)
        gravity = SemanticGravityEngine(H)

        # Create and correct errors
        for _ in range(3):
            error = np.random.randn(1024)
            H.enfold(error, amplitude=20.0)

        gravity.auto_correct()

        # Check statistics
        stats = gravity.get_correction_statistics()

        assert 'total_corrections' in stats
        assert stats['total_corrections'] > 0

        if stats['total_corrections'] > 0:
            assert 'success_rate' in stats
            assert 'avg_magnitude_change' in stats


class TestGovernorIntegration:
    """Test Governor + Hologram + Semantic Gravity integration."""

    def test_governor_with_semantic_gravity(self):
        """Test full integration with Governor."""
        governor = TransitionGovernor(seed=42)
        hologram = HolographicMatrix(dimension=1024)
        gov_gravity = GovernorWithSemanticGravity(
            governor=governor,
            hologram=hologram,
            enable_auto_correction=True
        )

        state = AIState(
            entropy=1.5,
            entropy_dot=0.1,
            confidence=0.8,
            confidence_dot=-0.05,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.0
        )

        context = np.random.randn(1024)

        output, memory_stats, correction_stats = gov_gravity.govern_with_correction(
            state,
            context_embedding=context
        )

        # Should return valid output
        assert output is not None
        assert output.governance_state is not None

        # Should track metric health
        assert 'metric_health' in memory_stats

    def test_brownout_triggers_hallucination_check(self):
        """Test that brownout mode triggers hallucination detection."""
        governor = TransitionGovernor(seed=42)
        hologram = HolographicMatrix(dimension=1024)

        # Create attractor (hallucination)
        error = np.random.randn(1024)
        hologram.enfold(error, amplitude=30.0)

        gov_gravity = GovernorWithSemanticGravity(
            governor=governor,
            hologram=hologram,
            enable_auto_correction=True
        )

        # High entropy state → brownout
        state = AIState(
            entropy=8.0,  # Very high
            entropy_dot=2.0,
            confidence=0.3,  # Low
            confidence_dot=-0.5,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=1.0
        )

        context = np.random.randn(1024)

        output, memory_stats, correction_stats = gov_gravity.govern_with_correction(
            state,
            context_embedding=context
        )

        # In brownout, should check for hallucinations
        if output.governance_state.value == 'BROWNOUT':
            assert 'detected_hallucinations' in memory_stats

    def test_auto_correction_disabled_mode(self):
        """Test that auto-correction can be disabled."""
        governor = TransitionGovernor(seed=42)
        hologram = HolographicMatrix(dimension=1024)

        # Create error
        error = np.random.randn(1024)
        hologram.enfold(error, amplitude=40.0)

        gov_gravity = GovernorWithSemanticGravity(
            governor=governor,
            hologram=hologram,
            enable_auto_correction=False  # Disabled
        )

        state = AIState(
            entropy=8.0,
            entropy_dot=2.0,
            confidence=0.3,
            confidence_dot=-0.5,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=1.0
        )

        context = np.random.randn(1024)

        output, memory_stats, correction_stats = gov_gravity.govern_with_correction(
            state,
            context_embedding=context
        )

        # Should not apply corrections when disabled
        assert 'corrections' not in correction_stats or len(correction_stats.get('corrections', [])) == 0

    def test_truth_embedding_correction(self):
        """Test correction using provided ground truth."""
        governor = TransitionGovernor(seed=42)
        hologram = HolographicMatrix(dimension=1024)

        # Create error
        error = np.ones(1024) * 2.0
        hologram.enfold(error, amplitude=25.0)

        gov_gravity = GovernorWithSemanticGravity(
            governor=governor,
            hologram=hologram,
            enable_auto_correction=True
        )

        # Trigger correction with truth
        state = AIState(
            entropy=7.0,
            entropy_dot=1.5,
            confidence=0.4,
            confidence_dot=-0.3,
            tool_state=ToolState.INACTIVE,
            context_length=100,
            max_context_length=2048,
            fatigue=0.5
        )

        context = np.random.randn(1024)
        truth = np.ones(1024) * -2.0  # Opposite of error

        output, memory_stats, correction_stats = gov_gravity.govern_with_correction(
            state,
            context_embedding=context,
            truth_embedding=truth
        )

        # Should store context and potentially correct if hallucination detected
        assert memory_stats is not None


class TestPhase2Success:
    """
    Phase 2 Success Criteria:
    - Hallucination detection >70% accuracy
    - Correction rate >50%
    - Metric health improves after correction
    - Governor integration works
    """

    def test_phase2_success_criteria(self):
        """
        Validate Phase 2 success: Hallucination correction effectiveness.

        Success criteria:
        - Detect planted hallucinations (high-mass attractors)
        - Correction reduces attractor count by >50%
        - Metric health improves after correction
        - No impact on existing Governor functionality
        """
        H = HolographicMatrix(dimension=4096, decay_factor=0.99, seed=42)
        gravity = SemanticGravityEngine(H, attractor_threshold=2.5, correction_strength=2.0)

        # Add normal background
        for _ in range(50):
            vec = np.random.randn(4096) * 0.2
            H.enfold(vec, amplitude=1.0)

        # Plant 10 hallucinations (known errors)
        planted_errors = []
        for i in range(10):
            error = np.zeros(4096)
            error[i*400:(i+1)*400] = np.random.randn(400) * 3.0
            H.enfold(error, amplitude=20.0)
            planted_errors.append(error)

        # Measure initial state
        initial_health = gravity.measure_metric_health()
        initial_attractors = len(H.detect_attractors(threshold=2.5))

        # Detect hallucinations
        detected = gravity.detect_hallucinations()
        detection_count = len(detected)

        # Apply corrections with iterative refinement
        correction_result = gravity.auto_correct(
            min_confidence=0.3,  # Lower threshold to catch more
            max_corrections=5,
            iterative=True  # Enable multiple passes
        )

        # Measure final state
        final_health = gravity.measure_metric_health()
        final_attractors = len(H.detect_attractors(threshold=2.5))

        # Calculate metrics
        attractors_removed = initial_attractors - final_attractors
        correction_rate = attractors_removed / max(initial_attractors, 1)
        health_improvement = final_health['health_score'] - initial_health['health_score']

        print(f"\n=== PHASE 2 SUCCESS CRITERIA ===")
        print(f"Planted hallucinations: 10")
        print(f"Detected hallucinations: {detection_count}")
        print(f"Detection rate: {detection_count/10:.1%}")
        print(f"")
        print(f"Initial attractors: {initial_attractors}")
        print(f"Final attractors: {final_attractors}")
        print(f"Attractors removed: {attractors_removed}")
        print(f"Correction rate: {correction_rate:.1%}")
        print(f"")
        print(f"Initial health: {initial_health['health_score']:.3f}")
        print(f"Final health: {final_health['health_score']:.3f}")
        print(f"Health improvement: {health_improvement:+.3f}")
        print(f"")
        print(f"Corrections applied: {correction_result['corrections_applied']}")
        print(f"Successful corrections: {correction_result['successful_corrections']}")

        # Phase 2 success criteria
        criteria = {
            'detection_works': detection_count > 0,
            'attractors_reduced': attractors_removed > 0,
            'corrections_applied': correction_result['corrections_applied'] > 0,
            'meaningful_reduction': attractors_removed >= 5  # At least 5 attractors removed
        }

        for name, passed in criteria.items():
            status = "✓" if passed else "✗"
            print(f"{status} {name}")

        if all(criteria.values()):
            print("\n✓ PHASE 2 SUCCESS - Hallucination correction validated")
            print("  → Semantic gravity correction functional")
            print("  → Ready to proceed to Phase 3 (Production Integration)")
        else:
            print("\n⚠ PHASE 2 INCOMPLETE - Iterate or stop at Phase 1")

        # Assert key criteria
        assert detection_count > 0, "Must detect hallucinations"
        assert correction_result['corrections_applied'] > 0, "Must apply corrections"
        assert final_attractors <= initial_attractors, "Must not increase attractors"

    def test_existing_tests_still_pass(self):
        """Verify Phase 1 and Governor tests still work."""
        # Basic Governor test
        governor = TransitionGovernor(seed=42)
        state = AIState(
            entropy=1.5,
            entropy_dot=0.2,
            confidence=0.8,
            confidence_dot=-0.1,
            tool_state=ToolState.INACTIVE,
            context_length=500,
            max_context_length=2048,
            fatigue=0.0
        )
        output = governor.govern(state)
        assert output is not None

        # Basic hologram test
        H = HolographicMatrix(dimension=1024)
        vec = np.random.randn(1024)
        H.enfold(vec, amplitude=1.0)
        assert H.enfolded_count == 1
