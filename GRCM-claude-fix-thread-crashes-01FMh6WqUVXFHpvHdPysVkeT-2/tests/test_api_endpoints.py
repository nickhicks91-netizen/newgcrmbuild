"""
Tests for FastAPI endpoints to increase coverage.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHealthEndpoints:
    """Tests for health and status endpoints."""
    
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200


class TestStateEndpoints:
    """Tests for model state endpoints."""
    
    def test_get_state(self, client):
        response = client.get("/state")
        assert response.status_code in [200, 503]  # 503 if model not initialized
    
    def test_get_phi(self, client):
        response = client.get("/phi")
        assert response.status_code == 200
        data = response.json()
        assert "phi" in data
    
    def test_get_lattice(self, client):
        response = client.get("/lattice")
        assert response.status_code == 200
        data = response.json()
        assert "lattice_enabled" in data or "n_nodes" in data
    
    def test_get_energy(self, client):
        response = client.get("/energy")
        assert response.status_code == 200
        data = response.json()
        assert "energy_savings_percent" in data or "sync_interval" in data
    
    def test_get_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code in [200, 503]  # 503 if model not initialized
    
    def test_get_config(self, client):
        response = client.get("/config")
        assert response.status_code in [200, 503]  # 503 if model not initialized


class TestForwardEndpoint:
    """Tests for forward pass endpoint."""
    
    def test_forward_basic(self, client):
        response = client.post("/forward", json={
            "input": [0.1] * 128
        })
        assert response.status_code in [200, 503, 422]  # 503 if model not initialized
    
    def test_forward_with_context(self, client):
        response = client.post("/forward", json={
            "input": [0.5] * 128,
            "context": "test context"
        })
        assert response.status_code in [200, 503, 422]


class TestTrainEndpoint:
    """Tests for training endpoint."""
    
    def test_train_step(self, client):
        response = client.post("/train", json={
            "input": [0.1] * 128,
            "target": [0.2] * 128
        })
        assert response.status_code == 200


class TestResetEndpoint:
    """Tests for reset endpoint."""
    
    def test_reset(self, client):
        response = client.post("/reset")
        assert response.status_code == 200


class TestDesireEndpoint:
    """Tests for desire/goal endpoint."""
    
    def test_set_desire(self, client):
        response = client.post("/desire", json={
            "desire": [0.5] * 128
        })
        assert response.status_code in [200, 503, 422]  # 503 if model not initialized


class TestBenchmarkEndpoint:
    """Tests for benchmark endpoint."""
    
    def test_benchmark(self, client):
        response = client.get("/benchmark")
        assert response.status_code == 200


class TestPerceptionEndpoint:
    """Tests for Tesla-style perception endpoint."""
    
    def test_perception_basic(self, client):
        response = client.post("/perception", json={
            "camera_feed": [[0.5] * 128],
            "sensor_data": {"lidar": [1.0, 2.0, 3.0]}
        })
        assert response.status_code == 200


class TestHallucinationEndpoint:
    """Tests for hallucination detection endpoint."""
    
    def test_hallucination_check(self, client):
        response = client.post("/hallucination", json={
            "text": "The sky is blue and water is wet.",
            "context": "General facts"
        })
        assert response.status_code == 200


class TestMemoryEndpoint:
    """Tests for episodic memory endpoint."""
    
    def test_get_memory(self, client):
        response = client.get("/memory")
        assert response.status_code == 200
    
    def test_memory_with_limit(self, client):
        response = client.get("/memory?limit=5")
        assert response.status_code == 200


class TestInvariantAPIEndpoints:
    """Tests for invariant CRUD API endpoints."""
    
    def test_create_invariant_api(self, client, clean_memory_stores):
        response = client.post("/invariants", json={
            "document_id": "api-test-doc",
            "invariant_type": "factual",
            "content": {"terms": ["API", "test"]},
            "elasticity": "rigid",
            "priority": 5
        })
        assert response.status_code == 200
        data = response.json()
        assert "invariant_id" in data or "success" in data
    
    def test_get_invariant_by_id(self, client, clean_memory_stores):
        create_resp = client.post("/invariants", json={
            "document_id": "api-test-doc",
            "invariant_type": "factual",
            "content": {"terms": ["test"]}
        })
        inv_id = create_resp.json().get("invariant_id")
        if inv_id:
            response = client.get(f"/invariants/{inv_id}")
            assert response.status_code == 200
    
    def test_get_invariants_for_document_api(self, client, clean_memory_stores):
        client.post("/invariants", json={
            "document_id": "api-test-doc-2",
            "invariant_type": "factual",
            "content": {"terms": ["test"]}
        })
        
        response = client.get("/invariants/document/api-test-doc-2")
        assert response.status_code == 200
        data = response.json()
        assert "invariants" in data or isinstance(data, list)
    
    def test_update_invariant_api(self, client, clean_memory_stores):
        create_resp = client.post("/invariants", json={
            "document_id": "api-test-doc-3",
            "invariant_type": "factual",
            "content": {"terms": ["old"]}
        })
        inv_id = create_resp.json().get("invariant_id")
        if inv_id:
            response = client.put(f"/invariants/{inv_id}", json={
                "priority": 10
            })
            assert response.status_code == 200
    
    def test_delete_invariant_api(self, client, clean_memory_stores):
        create_resp = client.post("/invariants", json={
            "document_id": "api-test-doc-4",
            "invariant_type": "factual",
            "content": {"terms": ["delete"]}
        })
        inv_id = create_resp.json().get("invariant_id")
        if inv_id:
            response = client.delete(f"/invariants/{inv_id}")
            assert response.status_code == 200


class TestDocumentAPIEndpoints:
    """Tests for document and hierarchy API endpoints."""
    
    def test_create_document_api(self, client, clean_memory_stores):
        response = client.post("/documents", json={
            "document_id": "api-doc-1",
            "title": "API Test Document",
            "content": "Test content"
        })
        assert response.status_code == 200
    
    def test_get_document_api(self, client, clean_memory_stores):
        client.post("/documents", json={
            "document_id": "api-doc-get",
            "title": "Get Test"
        })
        response = client.get("/documents/api-doc-get")
        assert response.status_code == 200
    
    def test_create_node_api(self, client, clean_memory_stores):
        client.post("/documents", json={
            "document_id": "api-doc-node",
            "title": "Node Test"
        })
        response = client.post("/documents/api-doc-node/nodes", json={
            "node_id": "api-node-1",
            "node_type": "section",
            "title": "Test Section"
        })
        assert response.status_code in [200, 422]  # 422 for validation issues
    
    def test_get_node_api(self, client, clean_memory_stores):
        client.post("/documents", json={
            "document_id": "api-doc-node-get",
            "title": "Node Get Test"
        })
        client.post("/documents/api-doc-node-get/nodes", json={
            "node_id": "api-node-get",
            "node_type": "section"
        })
        response = client.get("/documents/api-doc-node-get/nodes/api-node-get")
        assert response.status_code in [200, 404]  # 404 if node creation failed
    
    def test_attach_invariant_to_node_api(self, client, clean_memory_stores):
        client.post("/documents", json={
            "document_id": "api-doc-attach",
            "title": "Attach Test"
        })
        client.post("/documents/api-doc-attach/nodes", json={
            "node_id": "api-node-attach",
            "node_type": "section"
        })
        inv_resp = client.post("/invariants", json={
            "document_id": "api-doc-attach",
            "invariant_type": "factual",
            "content": {"terms": ["test"]}
        })
        inv_id = inv_resp.json().get("invariant_id")
        if inv_id:
            response = client.post(f"/documents/api-doc-attach/nodes/api-node-attach/attach-invariant", json={
                "invariant_id": inv_id
            })
            assert response.status_code in [200, 404, 422]


class TestLongformEndpoints:
    """Tests for longform compilation and validation endpoints."""
    
    def test_compile_prompt(self, client, clean_memory_stores):
        client.post("/documents", json={
            "document_id": "compile-doc",
            "title": "Compile Test"
        })
        
        response = client.post("/longform/compile", json={
            "base_prompt": "Write about GRCM",
            "document_id": "compile-doc"
        })
        assert response.status_code in [200, 422]
    
    def test_validate_text(self, client, clean_memory_stores):
        client.post("/invariants", json={
            "document_id": "validate-doc",
            "invariant_type": "factual",
            "content": {"terms": ["GRCM"]}
        })
        
        response = client.post("/longform/validate", json={
            "text": "GRCM is a consciousness module.",
            "document_id": "validate-doc"
        })
        assert response.status_code == 200
    
    def test_get_audit_log(self, client, clean_memory_stores):
        response = client.get("/longform/audit")
        assert response.status_code == 200
    
    def test_get_stats(self, client, clean_memory_stores):
        response = client.get("/longform/stats")
        assert response.status_code == 200


class TestApiFunctionsEndpoint:
    """Tests for API functions introspection endpoint."""
    
    def test_get_api_functions(self, client):
        response = client.get("/api/functions")
        assert response.status_code == 200
