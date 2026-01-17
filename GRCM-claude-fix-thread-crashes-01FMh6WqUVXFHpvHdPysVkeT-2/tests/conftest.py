"""
Pytest configuration and fixtures for GRCM tests.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clean_database():
    """Clear database tables before each test to prevent state leakage."""
    from main import get_db_connection
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM coherence_audit_log")
                cur.execute("DELETE FROM hierarchy_nodes")
                cur.execute("DELETE FROM invariants")
                cur.execute("DELETE FROM state_attractors")
                cur.execute("DELETE FROM episodic_memory")
                cur.execute("DELETE FROM memory_patterns")
            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            conn.close()
    yield


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from main import app
    return TestClient(app)


@pytest.fixture
def clean_memory_stores():
    """Clear in-memory stores before each test."""
    from main import _memory_invariants, _memory_hierarchy, _memory_audit, _memory_invariant_counter
    _memory_invariants.clear()
    _memory_hierarchy.clear()
    _memory_audit.clear()
    _memory_invariant_counter[0] = 0
    yield
    _memory_invariants.clear()
    _memory_hierarchy.clear()
    _memory_audit.clear()
    _memory_invariant_counter[0] = 0


@pytest.fixture
def sample_invariants():
    """Sample invariants for testing."""
    return [
        {
            "id": 1,
            "document_id": "test-doc",
            "invariant_type": "factual",
            "content": {"terms": ["GRCM", "consciousness"]},
            "elasticity": "rigid",
            "priority": 5,
            "is_active": True
        },
        {
            "id": 2,
            "document_id": "test-doc",
            "invariant_type": "terminological",
            "content": {"forbidden": ["magic"], "preferred": ["neural"]},
            "elasticity": "firm",
            "priority": 3,
            "is_active": True
        },
        {
            "id": 3,
            "document_id": "test-doc",
            "invariant_type": "tonal",
            "content": {"forbidden_patterns": ["lol", "omg"]},
            "elasticity": "flexible",
            "priority": 2,
            "is_active": True
        }
    ]


@pytest.fixture
def sample_node():
    """Sample hierarchy node for testing."""
    return {
        "node_id": "test-node-1",
        "document_id": "test-doc",
        "parent_id": None,
        "node_type": "document",
        "title": "Test Document",
        "content": "This is test content about GRCM.",
        "depth": 0,
        "position": 0,
        "inherited_invariants": [],
        "local_invariants": [],
        "metadata": {}
    }
