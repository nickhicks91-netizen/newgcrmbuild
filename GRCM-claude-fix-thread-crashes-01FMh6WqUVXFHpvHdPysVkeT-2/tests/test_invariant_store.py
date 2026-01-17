"""
Unit tests for InvariantStore - CRUD operations and edge cases.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import InvariantStore, _memory_invariants, _memory_invariant_counter


class TestInvariantStoreCreate:
    """Tests for InvariantStore.create()"""
    
    def test_create_basic_invariant(self, clean_memory_stores):
        inv_id = InvariantStore.create(
            document_id="doc-1",
            invariant_type="factual",
            content={"terms": ["test"]},
            scope="global",
            elasticity="rigid"
        )
        assert inv_id is not None
        assert inv_id > 0
    
    def test_create_with_node_id(self, clean_memory_stores):
        inv_id = InvariantStore.create(
            document_id="doc-1",
            invariant_type="tonal",
            content={"tone": "professional"},
            node_id="node-1",
            priority=5
        )
        assert inv_id is not None
        inv = InvariantStore.get(inv_id)
        assert inv["node_id"] == "node-1"
        assert inv["priority"] == 5
    
    def test_create_all_invariant_types(self, clean_memory_stores):
        types = ["factual", "stylistic", "tonal", "structural", 
                 "terminological", "logical", "temporal", "causal"]
        for inv_type in types:
            inv_id = InvariantStore.create(
                document_id="doc-1",
                invariant_type=inv_type,
                content={"test": True}
            )
            assert inv_id is not None
    
    def test_create_all_elasticity_levels(self, clean_memory_stores):
        levels = ["rigid", "firm", "flexible", "soft"]
        for elasticity in levels:
            inv_id = InvariantStore.create(
                document_id="doc-1",
                invariant_type="factual",
                content={"test": True},
                elasticity=elasticity
            )
            inv = InvariantStore.get(inv_id)
            assert inv["elasticity"] == elasticity


class TestInvariantStoreGet:
    """Tests for InvariantStore.get()"""
    
    def test_get_existing_invariant(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        inv = InvariantStore.get(inv_id)
        assert inv is not None
        assert inv["id"] == inv_id
        assert inv["document_id"] == "doc-1"
    
    def test_get_nonexistent_invariant(self, clean_memory_stores):
        inv = InvariantStore.get(99999)
        assert inv is None
    
    def test_get_preserves_content(self, clean_memory_stores):
        content = {"terms": ["GRCM", "consciousness"], "facts": ["AI safety"]}
        inv_id = InvariantStore.create("doc-1", "factual", content)
        inv = InvariantStore.get(inv_id)
        assert inv["content"] == content


class TestInvariantStoreGetForDocument:
    """Tests for InvariantStore.get_for_document()"""
    
    def test_get_document_invariants(self, clean_memory_stores):
        InvariantStore.create("doc-1", "factual", {"terms": ["a"]})
        InvariantStore.create("doc-1", "tonal", {"tone": "formal"})
        InvariantStore.create("doc-2", "factual", {"terms": ["b"]})
        
        invariants = InvariantStore.get_for_document("doc-1")
        assert len(invariants) == 2
        assert all(inv["document_id"] == "doc-1" for inv in invariants)
    
    def test_get_empty_document(self, clean_memory_stores):
        invariants = InvariantStore.get_for_document("nonexistent")
        assert invariants == []
    
    def test_excludes_inactive_by_default(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        InvariantStore.update(inv_id, {"is_active": False})
        
        invariants = InvariantStore.get_for_document("doc-1")
        assert len(invariants) == 0
    
    def test_includes_inactive_when_requested(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        InvariantStore.update(inv_id, {"is_active": False})
        
        invariants = InvariantStore.get_for_document("doc-1", include_inactive=True)
        assert len(invariants) == 1


class TestInvariantStoreUpdate:
    """Tests for InvariantStore.update()"""
    
    def test_update_priority(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]}, priority=1)
        success = InvariantStore.update(inv_id, {"priority": 10})
        assert success
        inv = InvariantStore.get(inv_id)
        assert inv["priority"] == 10
    
    def test_update_elasticity(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]}, elasticity="rigid")
        InvariantStore.update(inv_id, {"elasticity": "soft"})
        inv = InvariantStore.get(inv_id)
        assert inv["elasticity"] == "soft"
    
    def test_update_content(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["old"]})
        InvariantStore.update(inv_id, {"content": {"terms": ["new"]}})
        inv = InvariantStore.get(inv_id)
        assert inv["content"] == {"terms": ["new"]}
    
    def test_update_nonexistent_succeeds_silently(self, clean_memory_stores):
        # Database UPDATE returns success even when no rows affected (common pattern)
        success = InvariantStore.update(99999, {"priority": 5})
        assert success  # No error, just 0 rows updated
    
    def test_update_disallowed_field_ignored(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        InvariantStore.update(inv_id, {"document_id": "hacked"})
        inv = InvariantStore.get(inv_id)
        assert inv["document_id"] == "doc-1"


class TestInvariantStoreDelete:
    """Tests for InvariantStore.delete()"""
    
    def test_soft_delete(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        success = InvariantStore.delete(inv_id, soft_delete=True)
        assert success
        inv = InvariantStore.get(inv_id)
        assert inv["is_active"] == False
    
    def test_hard_delete(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        success = InvariantStore.delete(inv_id, soft_delete=False)
        assert success
        inv = InvariantStore.get(inv_id)
        assert inv is None
    
    def test_delete_nonexistent_succeeds_silently(self, clean_memory_stores):
        # Database DELETE returns success even when no rows affected (common pattern)
        success = InvariantStore.delete(99999)
        assert success  # No error, just 0 rows deleted


class TestElasticityViolation:
    """Tests for InvariantStore.check_elasticity_violation()"""
    
    def test_rigid_always_blocks(self, clean_memory_stores):
        invariant = {"elasticity": "rigid"}
        result = InvariantStore.check_elasticity_violation(invariant, 0.1)
        assert result["action"] == "block"
        assert result["requires_regeneration"] == True
    
    def test_firm_blocks_high_severity(self, clean_memory_stores):
        invariant = {"elasticity": "firm"}
        result = InvariantStore.check_elasticity_violation(invariant, 0.6)
        assert result["action"] == "block"
    
    def test_firm_warns_low_severity(self, clean_memory_stores):
        invariant = {"elasticity": "firm"}
        result = InvariantStore.check_elasticity_violation(invariant, 0.3)
        assert result["action"] == "warn"
    
    def test_flexible_warns_high_severity(self, clean_memory_stores):
        invariant = {"elasticity": "flexible"}
        result = InvariantStore.check_elasticity_violation(invariant, 0.8)
        assert result["action"] == "warn"
    
    def test_flexible_softens_low_severity(self, clean_memory_stores):
        invariant = {"elasticity": "flexible"}
        result = InvariantStore.check_elasticity_violation(invariant, 0.3)
        assert result["action"] == "soften"
    
    def test_soft_always_allows(self, clean_memory_stores):
        invariant = {"elasticity": "soft"}
        result = InvariantStore.check_elasticity_violation(invariant, 1.0)
        assert result["action"] == "allow"
        assert result["requires_regeneration"] == False
