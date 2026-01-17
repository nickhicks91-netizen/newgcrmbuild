"""
Tests for in-memory fallback mode without DATABASE_URL.
These tests mock the database connection to simulate no-database mode.
"""
import pytest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    InvariantStore, HierarchyBuilder, InvariantViolationGate,
    _memory_invariants, _memory_hierarchy, _memory_audit, _memory_invariant_counter
)


@pytest.fixture(autouse=True)
def disable_database():
    """Mock get_db_connection to return None, forcing in-memory mode."""
    with patch('main.get_db_connection', return_value=None):
        yield


class TestInvariantStoreMemoryFallback:
    """Test InvariantStore operations in memory-only mode."""
    
    def test_create_stores_in_memory(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        assert inv_id in _memory_invariants
        assert _memory_invariants[inv_id]["document_id"] == "doc-1"
    
    def test_get_retrieves_from_memory(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        inv = InvariantStore.get(inv_id)
        assert inv is not None
        assert inv == _memory_invariants[inv_id]
    
    def test_get_for_document_filters_memory(self, clean_memory_stores):
        InvariantStore.create("doc-1", "factual", {"terms": ["a"]})
        InvariantStore.create("doc-1", "tonal", {"tone": "formal"})
        InvariantStore.create("doc-2", "factual", {"terms": ["b"]})
        
        results = InvariantStore.get_for_document("doc-1")
        assert len(results) == 2
        assert all(r["document_id"] == "doc-1" for r in results)
    
    def test_update_modifies_memory(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["old"]}, priority=1)
        InvariantStore.update(inv_id, {"priority": 10})
        assert _memory_invariants[inv_id]["priority"] == 10
    
    def test_soft_delete_sets_inactive(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        InvariantStore.delete(inv_id, soft_delete=True)
        assert _memory_invariants[inv_id]["is_active"] == False
    
    def test_hard_delete_removes_from_memory(self, clean_memory_stores):
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        InvariantStore.delete(inv_id, soft_delete=False)
        assert inv_id not in _memory_invariants
    
    def test_counter_increments(self, clean_memory_stores):
        id1 = InvariantStore.create("doc-1", "factual", {"terms": ["a"]})
        id2 = InvariantStore.create("doc-1", "factual", {"terms": ["b"]})
        id3 = InvariantStore.create("doc-1", "factual", {"terms": ["c"]})
        assert id1 < id2 < id3


class TestHierarchyBuilderMemoryFallback:
    """Test HierarchyBuilder operations in memory-only mode."""
    
    def test_create_node_stores_in_memory(self, clean_memory_stores):
        HierarchyBuilder.create_node("node-1", "doc-1", "section", title="Test")
        assert "node-1" in _memory_hierarchy
        assert _memory_hierarchy["node-1"]["title"] == "Test"
    
    def test_get_node_retrieves_from_memory(self, clean_memory_stores):
        HierarchyBuilder.create_node("node-1", "doc-1", "section")
        node = HierarchyBuilder.get_node("node-1")
        assert node == _memory_hierarchy["node-1"]
    
    def test_get_document_tree_filters_memory(self, clean_memory_stores):
        HierarchyBuilder.create_node("n1", "doc-1", "section")
        HierarchyBuilder.create_node("n2", "doc-1", "section")
        HierarchyBuilder.create_node("n3", "doc-2", "section")
        
        tree = HierarchyBuilder.get_document_tree("doc-1")
        assert len(tree) == 2
    
    def test_get_children_filters_by_parent(self, clean_memory_stores):
        HierarchyBuilder.create_node("parent", "doc-1", "chapter")
        HierarchyBuilder.create_node("c1", "doc-1", "section", parent_id="parent", position=0)
        HierarchyBuilder.create_node("c2", "doc-1", "section", parent_id="parent", position=1)
        HierarchyBuilder.create_node("other", "doc-1", "section", parent_id=None)
        
        children = HierarchyBuilder.get_children("parent")
        assert len(children) == 2
    
    def test_inheritance_propagates_in_memory(self, clean_memory_stores):
        HierarchyBuilder.create_node("parent", "doc-1", "chapter")
        _memory_hierarchy["parent"]["inherited_invariants"] = [1, 2]
        _memory_hierarchy["parent"]["local_invariants"] = [3]
        
        HierarchyBuilder.create_node("child", "doc-1", "section", parent_id="parent")
        assert _memory_hierarchy["child"]["inherited_invariants"] == [1, 2, 3]
    
    def test_update_content_modifies_memory(self, clean_memory_stores):
        HierarchyBuilder.create_node("node-1", "doc-1", "paragraph", content="old")
        HierarchyBuilder.update_node_content("node-1", "new content")
        assert _memory_hierarchy["node-1"]["content"] == "new content"
    
    def test_delete_removes_from_memory(self, clean_memory_stores):
        HierarchyBuilder.create_node("node-1", "doc-1", "section")
        HierarchyBuilder.delete_node("node-1")
        assert "node-1" not in _memory_hierarchy
    
    def test_cascade_delete_removes_children(self, clean_memory_stores):
        HierarchyBuilder.create_node("p", "doc-1", "chapter")
        HierarchyBuilder.create_node("c1", "doc-1", "section", parent_id="p")
        HierarchyBuilder.create_node("c2", "doc-1", "section", parent_id="p")
        
        HierarchyBuilder.delete_node("p", cascade=True)
        assert "p" not in _memory_hierarchy
        assert "c1" not in _memory_hierarchy
        assert "c2" not in _memory_hierarchy


class TestAuditLogMemoryFallback:
    """Test audit logging in memory-only mode."""
    
    def test_audit_logs_to_memory(self, clean_memory_stores):
        InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        
        result = InvariantViolationGate.evaluate(
            text="Test text with test term",
            document_id="doc-1",
            session_id="session-1"
        )
        
        assert result["audit_id"] is not None
        assert len(_memory_audit) > 0
    
    def test_get_audit_log_retrieves_from_memory(self, clean_memory_stores):
        InvariantStore.create("doc-1", "factual", {"terms": ["a"]})
        
        InvariantViolationGate.evaluate("Text with a", "doc-1", session_id="s1")
        InvariantViolationGate.evaluate("Text with a", "doc-1", session_id="s2")
        
        all_entries = InvariantViolationGate.get_audit_log()
        assert len(all_entries) == 2
    
    def test_audit_log_filters_by_session(self, clean_memory_stores):
        InvariantStore.create("doc-1", "factual", {"terms": ["a"]})
        
        InvariantViolationGate.evaluate("Text a", "doc-1", session_id="session-A")
        InvariantViolationGate.evaluate("Text a", "doc-1", session_id="session-A")
        InvariantViolationGate.evaluate("Text a", "doc-1", session_id="session-B")
        
        entries = InvariantViolationGate.get_audit_log(session_id="session-A")
        assert len(entries) == 2
    
    def test_audit_log_filters_by_document(self, clean_memory_stores):
        InvariantStore.create("doc-1", "factual", {"terms": ["a"]})
        InvariantStore.create("doc-2", "factual", {"terms": ["b"]})
        
        InvariantViolationGate.evaluate("Text a", "doc-1")
        InvariantViolationGate.evaluate("Text b", "doc-2")
        
        entries = InvariantViolationGate.get_audit_log(document_id="doc-1")
        assert len(entries) == 1
        assert entries[0]["document_id"] == "doc-1"
    
    def test_audit_log_respects_limit(self, clean_memory_stores):
        InvariantStore.create("doc-1", "factual", {"terms": ["a"]})
        
        for i in range(10):
            InvariantViolationGate.evaluate("Text a", "doc-1", session_id=f"s{i}")
        
        entries = InvariantViolationGate.get_audit_log(limit=5)
        assert len(entries) == 5


class TestMemoryStoreIsolation:
    """Test that memory stores are properly isolated."""
    
    def test_stores_are_independent(self, clean_memory_stores):
        InvariantStore.create("doc-1", "factual", {"terms": ["a"]})
        HierarchyBuilder.create_node("node-1", "doc-1", "section")
        
        assert len(_memory_invariants) == 1
        assert len(_memory_hierarchy) == 1
        assert "doc-1" not in _memory_hierarchy
        assert "node-1" not in _memory_invariants
