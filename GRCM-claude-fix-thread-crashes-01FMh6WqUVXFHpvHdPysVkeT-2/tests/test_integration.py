"""
Integration tests - end-to-end workflow tests.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    InvariantStore, HierarchyBuilder, PromptCompiler, 
    CoherenceCritic, InvariantViolationGate
)


class TestFullCoherenceWorkflow:
    """End-to-end test of the coherence control workflow."""
    
    def test_complete_workflow(self, clean_memory_stores):
        doc_id = "integration-test-doc"
        HierarchyBuilder.create_document(doc_id, "Integration Test Document")
        chapter_id = HierarchyBuilder.create_node(
            "chapter-1", doc_id, "chapter",
            title="Chapter 1: Introduction",
            parent_id=f"doc_{doc_id}",
            depth=1
        )
        section_id = HierarchyBuilder.create_node(
            "section-1-1", doc_id, "section",
            title="Core Concepts",
            parent_id="chapter-1",
            content="GRCM provides AI safety through consciousness.",
            depth=2
        )
        
        inv1 = InvariantStore.create(
            doc_id, "factual",
            {"terms": ["GRCM", "consciousness", "safety"]},
            elasticity="rigid", priority=5
        )
        inv2 = InvariantStore.create(
            doc_id, "terminological",
            {"forbidden": ["magic", "supernatural"]},
            elasticity="firm", priority=3
        )
        
        compiled = PromptCompiler.compile_prompt(
            base_prompt="Explain how GRCM works.",
            document_id=doc_id,
            node_id="section-1-1",
            expansion_mode="guided"
        )
        
        assert compiled["injected_invariants"] == 2
        assert inv1 in compiled["invariant_ids"]
        assert inv2 in compiled["invariant_ids"]
        assert compiled["context_included"] == True
        assert "GRCM" in compiled["compiled_prompt"]
        
        good_text = "GRCM uses consciousness-based safety mechanisms to ensure AI alignment."
        result_pass = InvariantViolationGate.evaluate(
            text=good_text,
            document_id=doc_id,
            session_id="test-session"
        )
        assert result_pass["decision"] == "pass"
        assert result_pass["validation"]["valid"] == True
        assert result_pass["audit_id"] is not None
        
        bad_text = "This system works like magic to solve problems."
        result_fail = InvariantViolationGate.evaluate(
            text=bad_text,
            document_id=doc_id,
            session_id="test-session"
        )
        assert result_fail["decision"] == "regenerate"
        assert result_fail["validation"]["valid"] == False
        
        audit_entries = InvariantViolationGate.get_audit_log(session_id="test-session")
        assert len(audit_entries) == 2
        assert any(e["action_taken"] == "pass" for e in audit_entries)
        assert any(e["action_taken"] == "regenerate" for e in audit_entries)


class TestPromptCompilerIntegration:
    """Integration tests for PromptCompiler."""
    
    def test_expansion_modes(self, clean_memory_stores):
        doc_id = "prompt-test"
        HierarchyBuilder.create_document(doc_id, "Test Doc")
        InvariantStore.create(doc_id, "factual", {"terms": ["test"]}, priority=5)
        InvariantStore.create(doc_id, "tonal", {"tone": "formal"}, priority=1)
        
        strict = PromptCompiler.compile_prompt("Task", doc_id, expansion_mode="strict")
        assert "MUST" in strict["compiled_prompt"]
        assert strict["injected_invariants"] == 2
        
        guided = PromptCompiler.compile_prompt("Task", doc_id, expansion_mode="guided")
        assert "guidelines" in guided["compiled_prompt"].lower()
        
        light = PromptCompiler.compile_prompt("Task", doc_id, expansion_mode="light")
        assert light["injected_invariants"] == 1
        
        none_mode = PromptCompiler.compile_prompt("Task", doc_id, expansion_mode="none")
        assert none_mode["injected_invariants"] == 0
    
    def test_continuation_prompt(self, clean_memory_stores):
        doc_id = "cont-test"
        HierarchyBuilder.create_document(doc_id, "Test")
        HierarchyBuilder.create_node("node-1", doc_id, "section", content="Previous content here.")
        InvariantStore.create(doc_id, "factual", {"terms": ["AI"]})
        
        result = PromptCompiler.compile_for_continuation(doc_id, "node-1", "Some previous text...")
        assert "Continue" in result["compiled_prompt"]
        assert result["injected_invariants"] == 1


class TestInvariantPropagation:
    """Test invariant propagation through document hierarchy."""
    
    def test_invariants_propagate_down_tree(self, clean_memory_stores):
        doc_id = "prop-test"
        HierarchyBuilder.create_document(doc_id, "Propagation Test")
        HierarchyBuilder.create_node("ch1", doc_id, "chapter", parent_id=f"doc_{doc_id}")
        HierarchyBuilder.create_node("sec1", doc_id, "section", parent_id="ch1")
        HierarchyBuilder.create_node("para1", doc_id, "paragraph", parent_id="sec1")
        
        inv_id = InvariantStore.create(doc_id, "factual", {"terms": ["test"]})
        HierarchyBuilder.attach_invariant_to_node("ch1", inv_id)
        
        sec_node = HierarchyBuilder.get_node("sec1")
        para_node = HierarchyBuilder.get_node("para1")
        
        assert inv_id in sec_node["inherited_invariants"]
        assert inv_id in para_node["inherited_invariants"]


class TestPhiGateIntegration:
    """Test Phi score integration with coherence validation."""
    
    def test_low_phi_blocks_low_coherence(self, clean_memory_stores):
        doc_id = "phi-test"
        InvariantStore.create(doc_id, "factual", {"terms": ["required_term"]}, elasticity="rigid")
        
        result = InvariantViolationGate.evaluate(
            text="This text has no required terms.",
            document_id=doc_id,
            phi_score=0.2
        )
        
        assert result["phi_gate_passed"] == False
        assert result["decision"] == "block"
    
    def test_high_phi_allows_pass(self, clean_memory_stores):
        doc_id = "phi-test-2"
        InvariantStore.create(doc_id, "factual", {"terms": ["good"]}, elasticity="rigid")
        
        result = InvariantViolationGate.evaluate(
            text="This is a good text.",
            document_id=doc_id,
            phi_score=0.8
        )
        
        assert result["phi_gate_passed"] == True
        assert result["decision"] == "pass"


class TestElasticityBehavior:
    """Test different elasticity levels affect decisions correctly."""
    
    def test_rigid_violation_blocks(self, clean_memory_stores):
        doc_id = "elast-test"
        InvariantStore.create(doc_id, "factual", {"terms": ["must_have"]}, elasticity="rigid")
        
        result = InvariantViolationGate.evaluate("No required term.", doc_id)
        assert result["decision"] == "regenerate"
        assert result["validation"]["requires_regeneration"] == True
    
    def test_soft_violation_passes(self, clean_memory_stores):
        doc_id = "elast-test-2"
        InvariantStore.create(doc_id, "stylistic", {"avoid": ["basically"]}, elasticity="soft")
        
        result = InvariantViolationGate.evaluate("Basically, this is fine.", doc_id)
        assert result["decision"] == "pass"


class TestAPIEndpointIntegration:
    """Test API endpoints work together correctly."""
    
    def test_create_and_validate_flow(self, client, clean_memory_stores):
        doc_response = client.post("/documents", json={
            "document_id": "api-test",
            "title": "API Test Document"
        })
        assert doc_response.status_code == 200
        
        inv_response = client.post("/invariants", json={
            "document_id": "api-test",
            "invariant_type": "factual",
            "content": {"terms": ["API", "test"]},
            "elasticity": "rigid"
        })
        assert inv_response.status_code == 200
        
        validate_response = client.post("/longform/validate", json={
            "document_id": "api-test",
            "text": "This API test validates correctly.",
            "session_id": "api-session"
        })
        assert validate_response.status_code == 200
        data = validate_response.json()
        assert data["decision"] == "pass"
        
        audit_response = client.get("/longform/audit?session_id=api-session")
        assert audit_response.status_code == 200
        assert audit_response.json()["count"] >= 1
    
    def test_stats_reflect_operations(self, client, clean_memory_stores):
        client.post("/invariants", json={
            "document_id": "stats-test",
            "invariant_type": "factual",
            "content": {"terms": ["stat"]},
            "elasticity": "rigid"
        })
        client.post("/documents", json={
            "document_id": "stats-test",
            "title": "Stats Test"
        })
        
        stats = client.get("/longform/stats").json()
        assert stats["active_invariants"] >= 1
        assert stats["total_documents"] >= 1
