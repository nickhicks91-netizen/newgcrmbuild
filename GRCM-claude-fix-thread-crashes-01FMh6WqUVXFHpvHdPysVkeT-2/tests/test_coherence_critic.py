"""
Unit tests for CoherenceCritic - validation of all invariant types.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import CoherenceCritic, InvariantStore


class TestCoherenceCriticValidate:
    """Tests for CoherenceCritic.validate()"""
    
    def test_validate_empty_invariants(self):
        result = CoherenceCritic.validate("Any text here.", [])
        assert result["valid"] == True
        assert result["coherence_score"] == 0.0  # 0 passed / max(0 total, 1) = 0.0
        assert len(result["violations"]) == 0
    
    def test_validate_returns_phi_score(self):
        result = CoherenceCritic.validate("Test text", [], phi_score=0.75)
        assert result["phi_score"] == 0.75


class TestFactualInvariant:
    """Tests for factual invariant validation."""
    
    def test_factual_pass_all_terms(self):
        invariants = [{
            "id": 1,
            "invariant_type": "factual",
            "content": {"terms": ["GRCM", "consciousness"]},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("GRCM uses consciousness-based architecture.", invariants)
        assert result["valid"] == True
        assert result["coherence_score"] == 1.0
    
    def test_factual_fail_missing_term(self):
        invariants = [{
            "id": 1,
            "invariant_type": "factual",
            "content": {"terms": ["GRCM", "consciousness"]},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("This is a simple AI system.", invariants)
        assert result["valid"] == False
        assert len(result["violations"]) == 1
        assert "Missing required term" in result["violations"][0]["reason"]
    
    def test_factual_case_insensitive(self):
        invariants = [{
            "id": 1,
            "invariant_type": "factual",
            "content": {"terms": ["GRCM"]},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("The grcm system works well.", invariants)
        assert result["valid"] == True


class TestTerminologicalInvariant:
    """Tests for terminological invariant validation."""
    
    def test_terminological_pass_no_forbidden(self):
        invariants = [{
            "id": 1,
            "invariant_type": "terminological",
            "content": {"forbidden": ["magic", "supernatural"]},
            "elasticity": "firm"
        }]
        result = CoherenceCritic.validate("GRCM uses neural network principles.", invariants)
        assert result["valid"] == True
    
    def test_terminological_fail_forbidden_term(self):
        invariants = [{
            "id": 1,
            "invariant_type": "terminological",
            "content": {"forbidden": ["magic"]},
            "elasticity": "firm"
        }]
        result = CoherenceCritic.validate("It works like magic!", invariants)
        assert result["valid"] == False
        assert "Forbidden term used" in result["violations"][0]["reason"]
    
    def test_terminological_case_insensitive(self):
        invariants = [{
            "id": 1,
            "invariant_type": "terminological",
            "content": {"forbidden": ["MAGIC"]},
            "elasticity": "firm"
        }]
        result = CoherenceCritic.validate("It's like magic!", invariants)
        assert result["valid"] == False


class TestTonalInvariant:
    """Tests for tonal invariant validation."""
    
    def test_tonal_pass_professional(self):
        invariants = [{
            "id": 1,
            "invariant_type": "tonal",
            "content": {"forbidden_patterns": ["lol", "omg", "!!!"]},
            "elasticity": "flexible"
        }]
        result = CoherenceCritic.validate("GRCM provides enterprise-grade safety.", invariants)
        assert result["valid"] == True
    
    def test_tonal_fail_casual_pattern(self):
        invariants = [{
            "id": 1,
            "invariant_type": "tonal",
            "content": {"forbidden_patterns": ["lol"]},
            "elasticity": "flexible"
        }]
        result = CoherenceCritic.validate("This is amazing lol", invariants)
        assert len(result["warnings"]) == 1
        assert "Tone violation" in result["warnings"][0]["reason"]


class TestStructuralInvariant:
    """Tests for structural invariant validation."""
    
    def test_structural_pass_within_length(self):
        invariants = [{
            "id": 1,
            "invariant_type": "structural",
            "content": {"max_length": 100, "min_length": 10},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("This is a medium length text for testing.", invariants)
        assert result["valid"] == True
    
    def test_structural_fail_exceeds_max(self):
        invariants = [{
            "id": 1,
            "invariant_type": "structural",
            "content": {"max_length": 20},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("This text is way too long for the limit.", invariants)
        assert result["valid"] == False
        assert "Exceeds max length" in result["violations"][0]["reason"]
    
    def test_structural_fail_below_min(self):
        invariants = [{
            "id": 1,
            "invariant_type": "structural",
            "content": {"min_length": 100},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("Short.", invariants)
        assert result["valid"] == False
        assert "Below min length" in result["violations"][0]["reason"]


class TestLogicalInvariant:
    """Tests for logical invariant validation."""
    
    def test_logical_pass_no_contradictions(self):
        invariants = [{
            "id": 1,
            "invariant_type": "logical",
            "content": {"contradictions": [["hot", "cold"], ["up", "down"]]},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("The temperature is hot today.", invariants)
        assert result["valid"] == True
    
    def test_logical_fail_contradiction(self):
        invariants = [{
            "id": 1,
            "invariant_type": "logical",
            "content": {"contradictions": [["safe", "dangerous"]]},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("The system is both safe and dangerous.", invariants)
        assert result["valid"] == False
        assert "Logical contradiction" in result["violations"][0]["reason"]
        assert result["violations"][0]["severity"] == 0.9


class TestStylisticInvariant:
    """Tests for stylistic invariant validation."""
    
    def test_stylistic_pass_clean(self):
        invariants = [{
            "id": 1,
            "invariant_type": "stylistic",
            "content": {"avoid": ["very", "really", "basically"]},
            "elasticity": "flexible"
        }]
        result = CoherenceCritic.validate("GRCM provides excellent performance.", invariants)
        assert result["valid"] == True
    
    def test_stylistic_fail_filler_words(self):
        invariants = [{
            "id": 1,
            "invariant_type": "stylistic",
            "content": {"avoid": ["basically"]},
            "elasticity": "flexible"
        }]
        result = CoherenceCritic.validate("Basically, GRCM is a framework.", invariants)
        assert len(result["warnings"]) == 1


class TestDriftCalculation:
    """Tests for drift score calculation."""
    
    def test_drift_zero_when_all_terms_present(self):
        invariants = [{
            "id": 1,
            "invariant_type": "factual",
            "content": {"terms": ["GRCM", "AI"]},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("GRCM is an AI framework.", invariants)
        assert result["drift_score"] == 0.0
    
    def test_drift_one_when_all_terms_missing(self):
        invariants = [{
            "id": 1,
            "invariant_type": "factual",
            "content": {"terms": ["GRCM", "consciousness"]},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("This is a simple system.", invariants)
        assert result["drift_score"] == 1.0
    
    def test_drift_partial_terms(self):
        invariants = [{
            "id": 1,
            "invariant_type": "factual",
            "content": {"terms": ["GRCM", "AI", "neural", "safety"]},
            "elasticity": "rigid"
        }]
        result = CoherenceCritic.validate("GRCM provides AI capabilities.", invariants)
        assert result["drift_score"] == 0.5


class TestMultipleInvariants:
    """Tests for validating against multiple invariants."""
    
    def test_all_pass(self, sample_invariants):
        result = CoherenceCritic.validate(
            "GRCM uses consciousness-inspired neural architecture.",
            sample_invariants
        )
        assert result["valid"] == True
        assert result["coherence_score"] == 1.0
    
    def test_partial_pass(self, sample_invariants):
        result = CoherenceCritic.validate(
            "This system uses consciousness principles lol.",
            sample_invariants
        )
        assert result["coherence_score"] < 1.0
    
    def test_human_review_triggered(self):
        invariants = [
            {"id": i, "invariant_type": "factual", "content": {"terms": [f"term{i}"]}, "elasticity": "rigid"}
            for i in range(5)
        ]
        result = CoherenceCritic.validate("No terms match here.", invariants)
        assert result["requires_human_review"] == True
