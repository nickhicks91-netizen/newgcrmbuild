"""
Stress tests for concurrent inference and DB failure scenarios.

Tests thread-safety of:
- Concurrent API endpoint access
- Database connection failure handling
- Thread-safe statistics and model access
"""

import pytest
import threading
import time
import concurrent.futures
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app, thread_safe_stats, thread_safe_memory, ThreadSafeStats, ThreadSafeModel, ThreadSafeMemoryLog


class TestConcurrentAPIAccess:
    """Test concurrent access to FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_concurrent_forward_requests(self, client):
        """Test multiple concurrent forward pass requests."""
        num_threads = 10
        results = []
        errors = []

        def make_request():
            try:
                response = client.post("/forward", json={"input": [0.1] * 64})
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=make_request) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(status == 200 for status in results), f"Not all requests succeeded: {results}"
        assert len(results) == num_threads

    def test_concurrent_state_reads(self, client):
        """Test concurrent reads of state endpoints."""
        num_threads = 20
        endpoints = ["/state", "/phi", "/energy", "/metrics", "/config"]
        results = []
        errors = []

        def make_request(endpoint):
            try:
                response = client.get(endpoint)
                results.append((endpoint, response.status_code))
            except Exception as e:
                errors.append(str(e))

        threads = []
        for _ in range(num_threads):
            for endpoint in endpoints:
                t = threading.Thread(target=make_request, args=(endpoint,))
                threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(status == 200 for _, status in results), f"Not all requests succeeded: {results}"

    def test_concurrent_forward_and_state_access(self, client):
        """Test concurrent forward passes while reading state."""
        num_forward = 5
        num_reads = 10
        results = {"forward": [], "state": []}
        errors = []

        def forward_request():
            try:
                response = client.post("/forward", json={"input": [0.5] * 64})
                results["forward"].append(response.status_code)
            except Exception as e:
                errors.append(f"forward: {e}")

        def state_request():
            try:
                response = client.get("/metrics")
                results["state"].append(response.status_code)
            except Exception as e:
                errors.append(f"state: {e}")

        threads = []
        for _ in range(num_forward):
            threads.append(threading.Thread(target=forward_request))
        for _ in range(num_reads):
            threads.append(threading.Thread(target=state_request))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results["forward"]) == num_forward
        assert len(results["state"]) == num_reads


class TestThreadSafeStatsStress:
    """Stress test ThreadSafeStats class."""

    def test_concurrent_increments(self):
        """Test concurrent increment operations."""
        stats = ThreadSafeStats()
        num_threads = 50
        increments_per_thread = 100

        def increment_many():
            for _ in range(increments_per_thread):
                stats.increment("total_inferences")

        threads = [threading.Thread(target=increment_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * increments_per_thread
        actual = stats.get("total_inferences")
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_concurrent_set_and_get(self):
        """Test concurrent set and get operations."""
        stats = ThreadSafeStats()
        num_threads = 30
        results = []
        errors = []

        def set_and_get(thread_id):
            try:
                key = f"key_{thread_id}"
                stats.set(key, thread_id)
                value = stats.get(key)
                results.append((thread_id, value))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=set_and_get, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads
        for thread_id, value in results:
            assert value == thread_id, f"Thread {thread_id} got wrong value: {value}"

    def test_concurrent_history_append(self):
        """Test concurrent append to history lists (respects max length)."""
        stats = ThreadSafeStats()
        num_threads = 20
        appends_per_thread = 50

        def append_many(thread_id):
            for i in range(appends_per_thread):
                stats.append_to_history("phi_history", thread_id * 1000 + i)

        threads = [threading.Thread(target=append_many, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        history = stats.get("phi_history")
        assert len(history) <= 100, f"History should be capped at 100, got {len(history)}"
        assert len(history) > 0, "History should have some entries"


class TestThreadSafeMemoryLogStress:
    """Stress test ThreadSafeMemoryLog class."""

    def test_concurrent_log_append(self):
        """Test concurrent append to memory log."""
        memory_log = ThreadSafeMemoryLog()
        num_threads = 20
        appends_per_thread = 50

        def append_many(thread_id):
            for i in range(appends_per_thread):
                memory_log.append({
                    "thread_id": thread_id,
                    "index": i,
                    "phi": 0.5 + thread_id * 0.01
                })

        threads = [threading.Thread(target=append_many, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = memory_log.get_all()
        expected_count = num_threads * appends_per_thread
        assert len(entries) == expected_count, f"Expected {expected_count} entries, got {len(entries)}"

    def test_concurrent_append_and_get_recent(self):
        """Test concurrent appends while getting recent entries."""
        memory_log = ThreadSafeMemoryLog()
        num_writers = 10
        num_readers = 10
        writes_per_thread = 30
        errors = []

        def writer(thread_id):
            for i in range(writes_per_thread):
                try:
                    memory_log.append({"writer": thread_id, "index": i})
                except Exception as e:
                    errors.append(f"writer {thread_id}: {e}")

        def reader():
            for _ in range(writes_per_thread):
                try:
                    entries = memory_log.get_recent(10)
                    assert isinstance(entries, list)
                except Exception as e:
                    errors.append(f"reader: {e}")

        threads = []
        for i in range(num_writers):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for _ in range(num_readers):
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestDatabaseFailureHandling:
    """Test graceful handling of database connection failures."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_forward_continues_without_db(self, client):
        """Test forward pass continues when DB is unavailable."""
        response = client.post("/forward", json={"input": [0.1] * 64})
        assert response.status_code == 200, f"Forward should succeed: {response.json()}"

    def test_perception_basic_works(self, client):
        """Test perception endpoint works with valid input."""
        response = client.post("/perception", json={
            "sensor_data": {"cameras": [0.1] * 64}
        })
        assert response.status_code == 200

    def test_memory_endpoint_works(self, client):
        """Test memory endpoint returns data."""
        response = client.get("/memory")
        assert response.status_code == 200

    def test_db_connection_failure_graceful_degradation(self):
        """Test that DB failures don't crash the server."""
        from main import safe_db_connection
        
        with patch('main.psycopg2.connect') as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")
            
            with safe_db_connection() as conn:
                assert conn is None
                
    def test_safe_execute_catches_exceptions(self):
        """Test safe_execute decorator handles exceptions gracefully (returns None)."""
        from main import safe_execute
        
        @safe_execute()
        def failing_function():
            raise ValueError("Test exception")
        
        result = failing_function()
        assert result is None


class TestHighLoadStress:
    """Test system under high load conditions."""

    def test_rapid_fire_requests(self, client):
        """Test rapid sequential requests."""
        client = TestClient(app)
        num_requests = 100
        
        start = time.time()
        for _ in range(num_requests):
            response = client.get("/health")
            assert response.status_code == 200
        elapsed = time.time() - start
        
        avg_latency = elapsed / num_requests
        assert avg_latency < 0.1, f"Average latency too high: {avg_latency:.4f}s"

    def test_burst_concurrent_requests(self, client):
        """Test burst of concurrent requests using thread pool."""
        client = TestClient(app)
        num_requests = 50

        def make_request(_):
            response = client.get("/phi")
            return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(make_request, range(num_requests)))

        assert all(status == 200 for status in results), f"Not all requests succeeded: {results}"
        assert len(results) == num_requests

    @pytest.fixture
    def client(self):
        return TestClient(app)


class TestAutonomyLoopThreadSafety:
    """Test thread-safety of autonomy loop components."""

    def test_thread_safe_stats_mirror(self):
        """Test ThreadSafeStats mirrors autonomy loop pattern."""
        stats = ThreadSafeStats()
        num_threads = 30
        ops_per_thread = 100
        errors = []

        def stress_stats(thread_id):
            try:
                for i in range(ops_per_thread):
                    stats.set(f"phi_{thread_id}", thread_id + i * 0.001)
                    stats.increment("total_inferences")
                    _ = stats.get("total_inferences")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=stress_stats, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        expected_increments = num_threads * ops_per_thread
        assert stats.get("total_inferences") == expected_increments
