"""
Unit tests for HierarchyBuilder - node operations, cascading deletes, invariant propagation.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HierarchyBuilder, _memory_hierarchy


class TestHierarchyBuilderCreateNode:
    """Tests for HierarchyBuilder.create_node()"""
    
    def test_create_root_node(self, clean_memory_stores):
        node_id = HierarchyBuilder.create_node(
            node_id="root-1",
            document_id="doc-1",
            node_type="document",
            title="Test Document"
        )
        assert node_id == "root-1"
        node = HierarchyBuilder.get_node("root-1")
        assert node is not None
        assert node["title"] == "Test Document"
    
    def test_create_child_node(self, clean_memory_stores):
        HierarchyBuilder.create_node("parent", "doc-1", "chapter", title="Parent")
        child_id = HierarchyBuilder.create_node(
            node_id="child",
            document_id="doc-1",
            node_type="section",
            title="Child",
            parent_id="parent",
            depth=1
        )
        assert child_id == "child"
        child = HierarchyBuilder.get_node("child")
        assert child["parent_id"] == "parent"
        assert child["depth"] == 1
    
    def test_create_with_content(self, clean_memory_stores):
        HierarchyBuilder.create_node(
            node_id="content-node",
            document_id="doc-1",
            node_type="paragraph",
            content="This is the paragraph content."
        )
        node = HierarchyBuilder.get_node("content-node")
        assert node["content"] == "This is the paragraph content."
    
    def test_create_with_metadata(self, clean_memory_stores):
        HierarchyBuilder.create_node(
            node_id="meta-node",
            document_id="doc-1",
            node_type="section",
            metadata={"author": "Test", "version": 1}
        )
        node = HierarchyBuilder.get_node("meta-node")
        assert node["metadata"]["author"] == "Test"


class TestHierarchyBuilderGetNode:
    """Tests for HierarchyBuilder.get_node()"""
    
    def test_get_existing_node(self, clean_memory_stores):
        HierarchyBuilder.create_node("test-node", "doc-1", "section", title="Test")
        node = HierarchyBuilder.get_node("test-node")
        assert node is not None
        assert node["node_id"] == "test-node"
    
    def test_get_nonexistent_node(self, clean_memory_stores):
        node = HierarchyBuilder.get_node("nonexistent")
        assert node is None


class TestHierarchyBuilderGetDocumentTree:
    """Tests for HierarchyBuilder.get_document_tree()"""
    
    def test_get_full_tree(self, clean_memory_stores):
        HierarchyBuilder.create_node("root", "doc-1", "document", depth=0)
        HierarchyBuilder.create_node("ch1", "doc-1", "chapter", parent_id="root", depth=1, position=0)
        HierarchyBuilder.create_node("ch2", "doc-1", "chapter", parent_id="root", depth=1, position=1)
        HierarchyBuilder.create_node("sec1", "doc-1", "section", parent_id="ch1", depth=2, position=0)
        
        tree = HierarchyBuilder.get_document_tree("doc-1")
        assert len(tree) == 4
        assert tree[0]["depth"] == 0
        assert tree[1]["depth"] == 1
    
    def test_get_empty_tree(self, clean_memory_stores):
        tree = HierarchyBuilder.get_document_tree("nonexistent")
        assert tree == []
    
    def test_trees_are_document_specific(self, clean_memory_stores):
        HierarchyBuilder.create_node("node1", "doc-1", "document")
        HierarchyBuilder.create_node("node2", "doc-2", "document")
        
        tree1 = HierarchyBuilder.get_document_tree("doc-1")
        tree2 = HierarchyBuilder.get_document_tree("doc-2")
        assert len(tree1) == 1
        assert len(tree2) == 1
        assert tree1[0]["node_id"] == "node1"
        assert tree2[0]["node_id"] == "node2"


class TestHierarchyBuilderGetChildren:
    """Tests for HierarchyBuilder.get_children()"""
    
    def test_get_children(self, clean_memory_stores):
        HierarchyBuilder.create_node("parent", "doc-1", "chapter")
        HierarchyBuilder.create_node("child1", "doc-1", "section", parent_id="parent", position=0)
        HierarchyBuilder.create_node("child2", "doc-1", "section", parent_id="parent", position=1)
        HierarchyBuilder.create_node("child3", "doc-1", "section", parent_id="parent", position=2)
        
        children = HierarchyBuilder.get_children("parent")
        assert len(children) == 3
        assert children[0]["position"] == 0
        assert children[2]["position"] == 2
    
    def test_get_children_empty(self, clean_memory_stores):
        HierarchyBuilder.create_node("leaf", "doc-1", "paragraph")
        children = HierarchyBuilder.get_children("leaf")
        assert children == []


class TestHierarchyBuilderInvariantAttachment:
    """Tests for invariant attachment and propagation."""
    
    def test_attach_invariant_to_node(self, clean_memory_stores):
        from main import InvariantStore
        
        HierarchyBuilder.create_node("node1", "doc-1", "section")
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        
        success = HierarchyBuilder.attach_invariant_to_node("node1", inv_id)
        assert success
        
        node = HierarchyBuilder.get_node("node1")
        assert inv_id in node["local_invariants"]
    
    def test_attach_propagates_to_children(self, clean_memory_stores):
        from main import InvariantStore
        
        HierarchyBuilder.create_node("parent", "doc-1", "chapter")
        HierarchyBuilder.create_node("child", "doc-1", "section", parent_id="parent")
        inv_id = InvariantStore.create("doc-1", "factual", {"terms": ["test"]})
        
        HierarchyBuilder.attach_invariant_to_node("parent", inv_id)
        
        child = HierarchyBuilder.get_node("child")
        assert inv_id in child["inherited_invariants"]
    
    def test_attach_to_nonexistent_succeeds_silently(self, clean_memory_stores):
        # Database INSERT/UPDATE on nonexistent node doesn't fail
        success = HierarchyBuilder.attach_invariant_to_node("nonexistent", 1)
        assert success  # No error raised


class TestHierarchyBuilderUpdateAndDelete:
    """Tests for node update and delete operations."""
    
    def test_update_node_content(self, clean_memory_stores):
        HierarchyBuilder.create_node("node1", "doc-1", "paragraph", content="old")
        success = HierarchyBuilder.update_node_content("node1", "new content")
        assert success
        node = HierarchyBuilder.get_node("node1")
        assert node["content"] == "new content"
    
    def test_update_nonexistent_succeeds_silently(self, clean_memory_stores):
        # Database UPDATE returns success even when no rows affected
        success = HierarchyBuilder.update_node_content("nonexistent", "content")
        assert success  # No error, just 0 rows updated
    
    def test_delete_leaf_node(self, clean_memory_stores):
        HierarchyBuilder.create_node("leaf", "doc-1", "paragraph")
        success = HierarchyBuilder.delete_node("leaf")
        assert success
        node = HierarchyBuilder.get_node("leaf")
        assert node is None
    
    def test_delete_cascades_to_children(self, clean_memory_stores):
        HierarchyBuilder.create_node("parent", "doc-1", "chapter")
        HierarchyBuilder.create_node("child1", "doc-1", "section", parent_id="parent")
        HierarchyBuilder.create_node("child2", "doc-1", "section", parent_id="parent")
        HierarchyBuilder.create_node("grandchild", "doc-1", "paragraph", parent_id="child1")
        
        success = HierarchyBuilder.delete_node("parent", cascade=True)
        assert success
        
        assert HierarchyBuilder.get_node("parent") is None
        assert HierarchyBuilder.get_node("child1") is None
        assert HierarchyBuilder.get_node("child2") is None
        assert HierarchyBuilder.get_node("grandchild") is None
    
    def test_delete_without_cascade(self, clean_memory_stores):
        HierarchyBuilder.create_node("parent", "doc-1", "chapter")
        HierarchyBuilder.create_node("child", "doc-1", "section", parent_id="parent")
        
        success = HierarchyBuilder.delete_node("parent", cascade=False)
        assert success
        assert HierarchyBuilder.get_node("parent") is None
        assert HierarchyBuilder.get_node("child") is not None
    
    def test_delete_nonexistent_succeeds_silently(self, clean_memory_stores):
        # Database DELETE returns success even when no rows affected
        success = HierarchyBuilder.delete_node("nonexistent")
        assert success  # No error, just 0 rows deleted


class TestCreateDocument:
    """Tests for HierarchyBuilder.create_document()"""
    
    def test_create_document(self, clean_memory_stores):
        node_id = HierarchyBuilder.create_document("my-doc", "My Document", {"author": "Test"})
        assert node_id == "doc_my-doc"
        node = HierarchyBuilder.get_node("doc_my-doc")
        assert node["node_type"] == "document"
        assert node["title"] == "My Document"
        assert node["metadata"]["author"] == "Test"
