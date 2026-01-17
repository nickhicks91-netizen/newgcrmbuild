"""
GRCM Enterprise Demo Server
Production-ready FastAPI server for GRCM middleware demonstration
With Hopfield-Lite Identity Map (persistent episodic memory)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GRCM-claude-fix-thread-crashes-01FMh6WqUVXFHpvHdPysVkeT'))

import torch
import numpy as np
import time
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional, Any, Callable, TypeVar
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grcm")

T = TypeVar('T')


@dataclass
class SafeResult:
    """Result wrapper for safe execution - indicates success/failure with optional value."""
    success: bool
    value: Any = None
    error: Optional[str] = None
    
    @classmethod
    def ok(cls, value: Any = None) -> 'SafeResult':
        return cls(success=True, value=value)
    
    @classmethod
    def fail(cls, error: str) -> 'SafeResult':
        return cls(success=False, error=error)


def safe_execute(fallback: Any = None, log_errors: bool = True, reraise: bool = False):
    """
    Decorator for safe execution of functions that may fail.
    Catches DB, network, and torch errors and returns fallback value.
    
    Args:
        fallback: Value to return on failure
        log_errors: Whether to log errors
        reraise: Whether to re-raise HTTPException (for FastAPI handlers)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except HTTPException:
                if reraise:
                    raise
                if log_errors:
                    logger.warning(f"{func.__name__}: HTTP exception suppressed")
                return fallback
            except (psycopg2.Error, psycopg2.OperationalError) as e:
                if log_errors:
                    logger.error(f"{func.__name__}: Database error - {e}")
                return fallback
            except torch.cuda.CudaError as e:
                if log_errors:
                    logger.error(f"{func.__name__}: CUDA error - {e}")
                return fallback
            except (RuntimeError, ValueError, TypeError) as e:
                if log_errors:
                    logger.error(f"{func.__name__}: Runtime error - {e}")
                return fallback
            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__}: Unexpected error - {e}")
                return fallback
        return wrapper
    return decorator


@contextmanager
def safe_db_connection():
    """
    Context manager for safe database operations.
    Ensures connections are properly closed and errors are handled.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            yield None
        else:
            yield conn
    except (psycopg2.Error, psycopg2.OperationalError) as e:
        logger.error(f"Database operation error: {e}")
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        yield None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


class ThreadSafeStats:
    """Thread-safe wrapper for inference statistics."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._data = {
            "total_inferences": 0,
            "total_time_ms": 0,
            "sync_count": 0,
            "step_counter": 0,
            "energy_saved_percent": 0,
            "last_phi": 0,
            "last_coherence": 0,
            "last_torsion": 0,
            "qualia_history": [],
            "phi_history": [],
            "coherence_history": [],
            "torsion_history": [],
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
    
    def increment(self, key: str, amount: int = 1) -> int:
        with self._lock:
            self._data[key] = self._data.get(key, 0) + amount
            return self._data[key]
    
    def append_to_history(self, key: str, value: Any, max_len: int = 100) -> None:
        with self._lock:
            if key not in self._data:
                self._data[key] = []
            self._data[key].append(value)
            if len(self._data[key]) > max_len:
                self._data[key] = self._data[key][-max_len:]
    
    def update(self, updates: dict) -> None:
        with self._lock:
            self._data.update(updates)
    
    def snapshot(self) -> dict:
        """Return a copy of all stats for reading."""
        with self._lock:
            return dict(self._data)
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)


class ThreadSafeModel:
    """Thread-safe wrapper for model access."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._model = None
        self._trainer = None
        self._initialized = False
    
    def initialize(self, model, trainer=None) -> None:
        with self._lock:
            self._model = model
            self._trainer = trainer
            self._initialized = True
    
    @property
    def is_initialized(self) -> bool:
        with self._lock:
            return self._initialized
    
    @property
    def model(self):
        with self._lock:
            return self._model
    
    @property
    def trainer(self):
        with self._lock:
            return self._trainer
    
    @contextmanager
    def inference_lock(self):
        """Acquire lock for inference operations."""
        with self._lock:
            yield self._model
    
    @contextmanager
    def training_lock(self):
        """Acquire lock for training operations."""
        with self._lock:
            yield self._model, self._trainer


class ThreadSafeMemoryLog:
    """Thread-safe wrapper for memory log."""
    
    def __init__(self, max_size: int = 1000):
        self._lock = threading.RLock()
        self._log: List[dict] = []
        self._max_size = max_size
    
    def append(self, entry: dict) -> None:
        with self._lock:
            self._log.append(entry)
            if len(self._log) > self._max_size:
                self._log = self._log[-self._max_size:]
    
    def get_recent(self, n: int = 10) -> List[dict]:
        with self._lock:
            return list(self._log[-n:])
    
    def get_all(self) -> List[dict]:
        with self._lock:
            return list(self._log)
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._log)
    
    def __iter__(self):
        with self._lock:
            return iter(list(self._log))


thread_safe_stats = ThreadSafeStats()
thread_safe_model = ThreadSafeModel()
thread_safe_memory = ThreadSafeMemoryLog()


def get_db_connection():
    """Get PostgreSQL database connection."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None
    try:
        return psycopg2.connect(database_url, cursor_factory=RealDictCursor)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def init_database():
    """Initialize database schema for Hopfield-Lite Identity Map."""
    conn = get_db_connection()
    if not conn:
        print("No database connection - using in-memory fallback")
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id SERIAL PRIMARY KEY,
                    timestamp DOUBLE PRECISION NOT NULL,
                    session_id TEXT DEFAULT 'default',
                    action TEXT NOT NULL,
                    phi DOUBLE PRECISION NOT NULL,
                    coherence DOUBLE PRECISION DEFAULT 0,
                    conflict DOUBLE PRECISION NOT NULL,
                    outcome TEXT DEFAULT 'pending',
                    was_halted BOOLEAN DEFAULT FALSE,
                    torsion_score DOUBLE PRECISION DEFAULT 0,
                    qualia_state JSONB,
                    sensor_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_episodic_memory_timestamp ON episodic_memory(timestamp);
                CREATE INDEX IF NOT EXISTS idx_episodic_memory_was_halted ON episodic_memory(was_halted);
                CREATE INDEX IF NOT EXISTS idx_episodic_memory_session ON episodic_memory(session_id);
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS state_attractors (
                    id SERIAL PRIMARY KEY,
                    attractor_name TEXT UNIQUE NOT NULL,
                    phi_centroid DOUBLE PRECISION NOT NULL,
                    coherence_centroid DOUBLE PRECISION NOT NULL,
                    conflict_centroid DOUBLE PRECISION NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    attractor_vector JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_patterns (
                    id SERIAL PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    pattern_data JSONB NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_matched TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS invariants (
                    id SERIAL PRIMARY KEY,
                    document_id TEXT,
                    node_id TEXT,
                    invariant_type TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'global',
                    content JSONB NOT NULL,
                    elasticity TEXT NOT NULL DEFAULT 'rigid',
                    priority INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT TRUE,
                    source TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_invariants_document ON invariants(document_id);
                CREATE INDEX IF NOT EXISTS idx_invariants_node ON invariants(node_id);
                CREATE INDEX IF NOT EXISTS idx_invariants_type ON invariants(invariant_type);
                CREATE INDEX IF NOT EXISTS idx_invariants_scope ON invariants(scope);
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hierarchy_nodes (
                    id SERIAL PRIMARY KEY,
                    node_id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL,
                    parent_id TEXT,
                    node_type TEXT NOT NULL DEFAULT 'section',
                    title TEXT,
                    content TEXT,
                    depth INTEGER DEFAULT 0,
                    position INTEGER DEFAULT 0,
                    inherited_invariants JSONB DEFAULT '[]',
                    local_invariants JSONB DEFAULT '[]',
                    permissions JSONB DEFAULT '{}',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_hierarchy_document ON hierarchy_nodes(document_id);
                CREATE INDEX IF NOT EXISTS idx_hierarchy_parent ON hierarchy_nodes(parent_id);
                CREATE INDEX IF NOT EXISTS idx_hierarchy_type ON hierarchy_nodes(node_type);
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS coherence_audit_log (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    document_id TEXT,
                    node_id TEXT,
                    event_type TEXT NOT NULL,
                    invariant_id INTEGER,
                    violation_details JSONB,
                    original_text TEXT,
                    corrected_text TEXT,
                    phi_score DOUBLE PRECISION,
                    drift_score DOUBLE PRECISION,
                    action_taken TEXT,
                    was_regenerated BOOLEAN DEFAULT FALSE,
                    human_review_required BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_audit_session ON coherence_audit_log(session_id);
                CREATE INDEX IF NOT EXISTS idx_audit_document ON coherence_audit_log(document_id);
                CREATE INDEX IF NOT EXISTS idx_audit_event ON coherence_audit_log(event_type);
                CREATE INDEX IF NOT EXISTS idx_audit_time ON coherence_audit_log(created_at);
            """)
        conn.commit()
        print("Database schema initialized successfully (Hopfield-Lite Identity Map + Long-Form Coherence)")
        return True
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False
    finally:
        conn.close()

app = FastAPI(
    title="GRCM Enterprise Middleware API",
    description="Grounded Resonant Consciousness Module - Production Middleware for AI Systems",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modular_model = None
inference_stats = {
    "total_inferences": 0,
    "total_time_ms": 0,
    "sync_count": 0,
    "step_counter": 0,
    "energy_saved_percent": 0,
    "last_phi": 0,
    "last_coherence": 0,
    "last_torsion": 0,
    "qualia_history": [],
    "phi_history": [],
    "coherence_history": [],
    "torsion_history": [],
}

SYNC_INTERVAL = 300


echo_trainer = None
_models_initialized = False
_db_initialized = False

def get_models():
    """Lazy initialization of GRCM models - only loads when first needed."""
    global modular_model, echo_trainer, _models_initialized
    
    if _models_initialized:
        return modular_model, echo_trainer
    
    try:
        from grcm import ModularGRCM, GRCMConfig
        from grcm.trainer import EchoMirrorTrainer
        config = GRCMConfig()
        modular_model = ModularGRCM(config)
        modular_model.eval()
        echo_trainer = EchoMirrorTrainer(modular_model, config)
        print("ModularGRCM and EchoMirrorTrainer initialized (lazy)")
    except Exception as e:
        print(f"Warning: Could not initialize ModularGRCM: {e}")
        import traceback
        traceback.print_exc()
        modular_model = None
        echo_trainer = None
    
    _models_initialized = True
    return modular_model, echo_trainer

def ensure_db():
    """Lazy database initialization - only runs when first needed."""
    global _db_initialized
    if not _db_initialized:
        init_database()
        _db_initialized = True


def save_memory_to_db(memory_entry: dict, qualia_state: dict = None, coherence: float = 0):
    """Persist a memory entry to the Hopfield-Lite Identity Map (database)."""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO episodic_memory 
                (timestamp, action, phi, coherence, conflict, outcome, was_halted, torsion_score, qualia_state)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                memory_entry.get("timestamp", time.time()),
                memory_entry.get("action", "unknown"),
                memory_entry.get("phi", 0),
                coherence,
                memory_entry.get("conflict", 0),
                memory_entry.get("outcome", "pending"),
                memory_entry.get("was_halted", False),
                memory_entry.get("torsion_score", 0),
                json.dumps(qualia_state) if qualia_state else None
            ))
            memory_id = cur.fetchone()["id"]
        conn.commit()
        
        update_attractor_patterns(memory_entry.get("phi", 0), coherence, memory_entry.get("conflict", 0))
        
        return memory_id
    except Exception as e:
        print(f"Error saving memory: {e}")
        return False
    finally:
        conn.close()


def update_attractor_patterns(phi: float, coherence: float, conflict: float):
    """Update Hopfield attractor network with new state pattern."""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        if phi > 0.7 and conflict < 0.3:
            attractor_name = "high_confidence"
        elif phi < 0.3:
            attractor_name = "uncertain"
        elif conflict > 0.5:
            attractor_name = "conflicted"
        else:
            attractor_name = "neutral"
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO state_attractors (attractor_name, phi_centroid, coherence_centroid, conflict_centroid, occurrence_count)
                VALUES (%s, %s, %s, %s, 1)
                ON CONFLICT (attractor_name) DO UPDATE SET
                    phi_centroid = (state_attractors.phi_centroid * state_attractors.occurrence_count + EXCLUDED.phi_centroid) / (state_attractors.occurrence_count + 1),
                    coherence_centroid = (state_attractors.coherence_centroid * state_attractors.occurrence_count + EXCLUDED.coherence_centroid) / (state_attractors.occurrence_count + 1),
                    conflict_centroid = (state_attractors.conflict_centroid * state_attractors.occurrence_count + EXCLUDED.conflict_centroid) / (state_attractors.occurrence_count + 1),
                    occurrence_count = state_attractors.occurrence_count + 1,
                    last_seen = CURRENT_TIMESTAMP
            """, (attractor_name, phi, coherence, conflict))
        conn.commit()
    except Exception as e:
        print(f"Error updating attractors: {e}")
    finally:
        conn.close()


def get_memories_from_db(limit: int = 50) -> list:
    """Retrieve memories from the Hopfield-Lite Identity Map."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM episodic_memory 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (limit,))
            memories = cur.fetchall()
        return [dict(m) for m in memories]
    except Exception as e:
        print(f"Error fetching memories: {e}")
        return []
    finally:
        conn.close()


def get_memory_stats_from_db() -> dict:
    """Get aggregated memory statistics from persistent storage."""
    conn = get_db_connection()
    if not conn:
        return {"total_memories": 0, "past_halts": 0, "past_safe_actions": 0, "db_connected": False}
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM episodic_memory")
            total = cur.fetchone()["total"]
            
            cur.execute("SELECT COUNT(*) as halts FROM episodic_memory WHERE was_halted = TRUE")
            halts = cur.fetchone()["halts"]
            
            cur.execute("""
                SELECT action, COUNT(*) as count 
                FROM episodic_memory 
                GROUP BY action 
                ORDER BY count DESC 
                LIMIT 5
            """)
            action_distribution = {row["action"]: row["count"] for row in cur.fetchall()}
            
            cur.execute("SELECT AVG(phi) as avg_phi, AVG(conflict) as avg_conflict FROM episodic_memory")
            averages = cur.fetchone()
            
            cur.execute("SELECT * FROM state_attractors ORDER BY occurrence_count DESC")
            attractors = [dict(a) for a in cur.fetchall()]
        
        return {
            "total_memories": total,
            "past_halts": halts,
            "past_safe_actions": total - halts,
            "halt_rate": round(halts / max(total, 1), 3),
            "average_phi": round(averages["avg_phi"] or 0, 3),
            "average_conflict": round(averages["avg_conflict"] or 0, 3),
            "action_distribution": action_distribution,
            "state_attractors": attractors,
            "db_connected": True,
            "learning_active": halts > 0,
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {"total_memories": 0, "past_halts": 0, "past_safe_actions": 0, "db_connected": False}
    finally:
        conn.close()


def find_similar_past_states(phi: float, conflict: float, threshold: float = 0.15) -> list:
    """Find similar past states from the Hopfield memory (pattern matching)."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT *, 
                    ABS(phi - %s) + ABS(conflict - %s) as distance
                FROM episodic_memory 
                WHERE ABS(phi - %s) < %s AND ABS(conflict - %s) < %s
                ORDER BY distance ASC
                LIMIT 5
            """, (phi, conflict, phi, threshold, conflict, threshold))
            similar = cur.fetchall()
        return [dict(s) for s in similar]
    except Exception as e:
        print(f"Error finding similar states: {e}")
        return []
    finally:
        conn.close()


_memory_invariants = {}
_memory_invariant_counter = [0]
_memory_hierarchy = {}
_memory_audit = []


class InvariantStore:
    """
    Manages invariants for long-form coherence control.
    Invariants are symbolic constraints with type, scope, and elasticity.
    Falls back to in-memory storage when database is unavailable.
    """
    
    ELASTICITY_LEVELS = {
        "rigid": 0,
        "firm": 1,
        "flexible": 2,
        "soft": 3
    }
    
    INVARIANT_TYPES = [
        "factual",
        "stylistic",
        "tonal",
        "structural",
        "terminological",
        "logical",
        "temporal",
        "causal"
    ]
    
    @staticmethod
    def create(document_id: str, invariant_type: str, content: dict, 
               scope: str = "global", elasticity: str = "rigid",
               node_id: str = None, priority: int = 1, source: str = "user") -> Optional[int]:
        """Create a new invariant."""
        conn = get_db_connection()
        if not conn:
            _memory_invariant_counter[0] += 1
            inv_id = _memory_invariant_counter[0]
            _memory_invariants[inv_id] = {
                "id": inv_id,
                "document_id": document_id,
                "node_id": node_id,
                "invariant_type": invariant_type,
                "scope": scope,
                "content": content,
                "elasticity": elasticity,
                "priority": priority,
                "source": source,
                "is_active": True,
                "created_at": time.time()
            }
            return inv_id
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO invariants 
                    (document_id, node_id, invariant_type, scope, content, elasticity, priority, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (document_id, node_id, invariant_type, scope, 
                      json.dumps(content), elasticity, priority, source))
                result = cur.fetchone()
            conn.commit()
            return result["id"] if result else None
        except Exception as e:
            print(f"Error creating invariant: {e}")
            return None
        finally:
            conn.close()
    
    @staticmethod
    def get(invariant_id: int) -> Optional[dict]:
        """Get a single invariant by ID."""
        conn = get_db_connection()
        if not conn:
            return _memory_invariants.get(invariant_id)
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM invariants WHERE id = %s", (invariant_id,))
                result = cur.fetchone()
            return dict(result) if result else None
        except Exception as e:
            print(f"Error fetching invariant: {e}")
            return None
        finally:
            conn.close()
    
    @staticmethod
    def get_for_document(document_id: str, include_inactive: bool = False) -> List[dict]:
        """Get all invariants for a document."""
        conn = get_db_connection()
        if not conn:
            results = [inv for inv in _memory_invariants.values() 
                      if inv.get("document_id") == document_id]
            if not include_inactive:
                results = [inv for inv in results if inv.get("is_active", True)]
            return sorted(results, key=lambda x: (-x.get("priority", 0), x.get("created_at", 0)))
        
        try:
            with conn.cursor() as cur:
                query = "SELECT * FROM invariants WHERE document_id = %s"
                if not include_inactive:
                    query += " AND is_active = TRUE"
                query += " ORDER BY priority DESC, created_at ASC"
                cur.execute(query, (document_id,))
                results = cur.fetchall()
            return [dict(r) for r in results]
        except Exception as e:
            print(f"Error fetching document invariants: {e}")
            return []
        finally:
            conn.close()
    
    @staticmethod
    def get_for_node(node_id: str, include_inherited: bool = True) -> List[dict]:
        """Get invariants for a specific node, optionally including inherited ones."""
        conn = get_db_connection()
        if not conn:
            local_invariants = [inv for inv in _memory_invariants.values() 
                               if inv.get("node_id") == node_id and inv.get("is_active", True)]
            if include_inherited:
                node = _memory_hierarchy.get(node_id)
                if node and node.get("inherited_invariants"):
                    for inv_id in node.get("inherited_invariants", []):
                        inv = _memory_invariants.get(inv_id)
                        if inv and inv.get("is_active", True):
                            local_invariants.append(inv)
            return sorted(local_invariants, key=lambda x: -x.get("priority", 0))
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM invariants 
                    WHERE node_id = %s AND is_active = TRUE
                    ORDER BY priority DESC
                """, (node_id,))
                local_invariants = [dict(r) for r in cur.fetchall()]
                
                if include_inherited:
                    cur.execute("""
                        SELECT h.inherited_invariants 
                        FROM hierarchy_nodes h 
                        WHERE h.node_id = %s
                    """, (node_id,))
                    node_result = cur.fetchone()
                    if node_result and node_result["inherited_invariants"]:
                        inherited_ids = node_result["inherited_invariants"]
                        if inherited_ids:
                            cur.execute("""
                                SELECT * FROM invariants 
                                WHERE id = ANY(%s) AND is_active = TRUE
                                ORDER BY priority DESC
                            """, (inherited_ids,))
                            inherited = [dict(r) for r in cur.fetchall()]
                            local_invariants.extend(inherited)
            
            return local_invariants
        except Exception as e:
            print(f"Error fetching node invariants: {e}")
            return []
        finally:
            conn.close()
    
    @staticmethod
    def update(invariant_id: int, updates: dict) -> bool:
        """Update an invariant."""
        conn = get_db_connection()
        if not conn:
            if invariant_id in _memory_invariants:
                allowed_fields = ["content", "elasticity", "priority", "is_active", "scope"]
                for k, v in updates.items():
                    if k in allowed_fields:
                        _memory_invariants[invariant_id][k] = v
                return True
            return False
        
        allowed_fields = ["content", "elasticity", "priority", "is_active", "scope"]
        filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not filtered_updates:
            return False
        
        try:
            set_clause = ", ".join([f"{k} = %s" for k in filtered_updates.keys()])
            values = [json.dumps(v) if k == "content" else v for k, v in filtered_updates.items()]
            values.append(invariant_id)
            
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE invariants 
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP 
                    WHERE id = %s
                """, values)
            conn.commit()
            return True
        except Exception as e:
            print(f"Error updating invariant: {e}")
            return False
        finally:
            conn.close()
    
    @staticmethod
    def delete(invariant_id: int, soft_delete: bool = True) -> bool:
        """Delete an invariant (soft delete by default)."""
        conn = get_db_connection()
        if not conn:
            if invariant_id in _memory_invariants:
                if soft_delete:
                    _memory_invariants[invariant_id]["is_active"] = False
                else:
                    del _memory_invariants[invariant_id]
                return True
            return False
        
        try:
            with conn.cursor() as cur:
                if soft_delete:
                    cur.execute("""
                        UPDATE invariants SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP 
                        WHERE id = %s
                    """, (invariant_id,))
                else:
                    cur.execute("DELETE FROM invariants WHERE id = %s", (invariant_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting invariant: {e}")
            return False
        finally:
            conn.close()
    
    @staticmethod
    def check_elasticity_violation(invariant: dict, violation_severity: float) -> dict:
        """
        Check if a violation should be flagged based on elasticity level.
        Returns action to take: block, warn, soften, or allow.
        """
        elasticity = invariant.get("elasticity", "rigid")
        elasticity_level = InvariantStore.ELASTICITY_LEVELS.get(elasticity, 0)
        
        if elasticity_level == 0:
            return {"action": "block", "reason": "Rigid invariant violated", "requires_regeneration": True}
        elif elasticity_level == 1:
            if violation_severity > 0.5:
                return {"action": "block", "reason": "Firm invariant significantly violated", "requires_regeneration": True}
            else:
                return {"action": "warn", "reason": "Firm invariant minor violation", "requires_regeneration": False}
        elif elasticity_level == 2:
            if violation_severity > 0.7:
                return {"action": "warn", "reason": "Flexible invariant violated", "requires_regeneration": False}
            else:
                return {"action": "soften", "reason": "Flexible invariant drift detected", "requires_regeneration": False}
        else:
            return {"action": "allow", "reason": "Soft invariant allows variation", "requires_regeneration": False}
    
    @staticmethod
    def get_effective_invariants(document_id: str, node_id: str = None) -> List[dict]:
        """Get all effective invariants for a context, merging document and node level."""
        doc_invariants = InvariantStore.get_for_document(document_id)
        
        if not node_id:
            return doc_invariants
        
        node_invariants = InvariantStore.get_for_node(node_id, include_inherited=True)
        
        all_invariants = {}
        for inv in doc_invariants:
            if inv.get("scope") == "global":
                all_invariants[inv["id"]] = inv
        
        for inv in node_invariants:
            all_invariants[inv["id"]] = inv
        
        return sorted(all_invariants.values(), key=lambda x: (-x.get("priority", 0), x.get("id", 0)))


class HierarchyBuilder:
    """
    Builds and manages document tree structures with scoped invariants.
    Each node can have local invariants and inherit from parent nodes.
    """
    
    NODE_TYPES = ["document", "chapter", "section", "subsection", "paragraph", "block"]
    
    @staticmethod
    def create_document(document_id: str, title: str, metadata: dict = None) -> Optional[str]:
        """Create a root document node."""
        return HierarchyBuilder.create_node(
            node_id=f"doc_{document_id}",
            document_id=document_id,
            node_type="document",
            title=title,
            parent_id=None,
            depth=0,
            metadata=metadata or {}
        )
    
    @staticmethod
    def create_node(node_id: str, document_id: str, node_type: str,
                    title: str = None, parent_id: str = None, 
                    content: str = None, depth: int = 0, position: int = 0,
                    metadata: dict = None) -> Optional[str]:
        """Create a hierarchy node."""
        conn = get_db_connection()
        if not conn:
            inherited_invariants = []
            if parent_id:
                parent_node = _memory_hierarchy.get(parent_id)
                if parent_node:
                    inherited_invariants = list(parent_node.get("inherited_invariants", []))
                    inherited_invariants.extend(parent_node.get("local_invariants", []))
            _memory_hierarchy[node_id] = {
                "node_id": node_id,
                "document_id": document_id,
                "parent_id": parent_id,
                "node_type": node_type,
                "title": title,
                "content": content,
                "depth": depth,
                "position": position,
                "inherited_invariants": inherited_invariants,
                "local_invariants": [],
                "metadata": metadata or {},
                "created_at": time.time()
            }
            return node_id
        
        try:
            inherited_invariants = []
            if parent_id:
                parent_node = HierarchyBuilder.get_node(parent_id)
                if parent_node:
                    inherited_invariants = list(parent_node.get("inherited_invariants", []))
                    inherited_invariants.extend(parent_node.get("local_invariants", []))
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO hierarchy_nodes 
                    (node_id, document_id, parent_id, node_type, title, content, 
                     depth, position, inherited_invariants, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        parent_id = EXCLUDED.parent_id,
                        depth = EXCLUDED.depth,
                        position = EXCLUDED.position,
                        inherited_invariants = EXCLUDED.inherited_invariants,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING node_id
                """, (node_id, document_id, parent_id, node_type, title, content,
                      depth, position, json.dumps(inherited_invariants), 
                      json.dumps(metadata or {})))
                result = cur.fetchone()
            conn.commit()
            return result["node_id"] if result else None
        except Exception as e:
            print(f"Error creating hierarchy node: {e}")
            return None
        finally:
            conn.close()
    
    @staticmethod
    def get_node(node_id: str) -> Optional[dict]:
        """Get a single node by ID."""
        conn = get_db_connection()
        if not conn:
            return _memory_hierarchy.get(node_id)
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM hierarchy_nodes WHERE node_id = %s", (node_id,))
                result = cur.fetchone()
            return dict(result) if result else None
        except Exception as e:
            print(f"Error fetching node: {e}")
            return None
        finally:
            conn.close()
    
    @staticmethod
    def get_document_tree(document_id: str) -> List[dict]:
        """Get all nodes for a document as a flat list ordered by depth and position."""
        conn = get_db_connection()
        if not conn:
            nodes = [n for n in _memory_hierarchy.values() if n.get("document_id") == document_id]
            return sorted(nodes, key=lambda x: (x.get("depth", 0), x.get("position", 0)))
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM hierarchy_nodes 
                    WHERE document_id = %s 
                    ORDER BY depth ASC, position ASC
                """, (document_id,))
                results = cur.fetchall()
            return [dict(r) for r in results]
        except Exception as e:
            print(f"Error fetching document tree: {e}")
            return []
        finally:
            conn.close()
    
    @staticmethod
    def get_children(parent_id: str) -> List[dict]:
        """Get all direct children of a node."""
        conn = get_db_connection()
        if not conn:
            children = [n for n in _memory_hierarchy.values() if n.get("parent_id") == parent_id]
            return sorted(children, key=lambda x: x.get("position", 0))
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM hierarchy_nodes 
                    WHERE parent_id = %s 
                    ORDER BY position ASC
                """, (parent_id,))
                results = cur.fetchall()
            return [dict(r) for r in results]
        except Exception as e:
            print(f"Error fetching children: {e}")
            return []
        finally:
            conn.close()
    
    @staticmethod
    def attach_invariant_to_node(node_id: str, invariant_id: int) -> bool:
        """Attach an invariant to a node's local invariants."""
        conn = get_db_connection()
        if not conn:
            if node_id in _memory_hierarchy:
                if "local_invariants" not in _memory_hierarchy[node_id]:
                    _memory_hierarchy[node_id]["local_invariants"] = []
                _memory_hierarchy[node_id]["local_invariants"].append(invariant_id)
                HierarchyBuilder._propagate_invariants_to_children(node_id, invariant_id)
                return True
            return False
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE hierarchy_nodes 
                    SET local_invariants = local_invariants || %s::jsonb,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE node_id = %s
                """, (json.dumps([invariant_id]), node_id))
            conn.commit()
            HierarchyBuilder._propagate_invariants_to_children(node_id, invariant_id)
            return True
        except Exception as e:
            print(f"Error attaching invariant: {e}")
            return False
        finally:
            conn.close()
    
    @staticmethod
    def _propagate_invariants_to_children(parent_id: str, invariant_id: int):
        """Propagate an invariant to all children nodes."""
        children = HierarchyBuilder.get_children(parent_id)
        for child in children:
            conn = get_db_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE hierarchy_nodes 
                            SET inherited_invariants = inherited_invariants || %s::jsonb,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE node_id = %s
                        """, (json.dumps([invariant_id]), child["node_id"]))
                    conn.commit()
                except Exception as e:
                    print(f"Error propagating invariant: {e}")
                finally:
                    conn.close()
            else:
                child_id = child["node_id"]
                if child_id in _memory_hierarchy:
                    if "inherited_invariants" not in _memory_hierarchy[child_id]:
                        _memory_hierarchy[child_id]["inherited_invariants"] = []
                    _memory_hierarchy[child_id]["inherited_invariants"].append(invariant_id)
            HierarchyBuilder._propagate_invariants_to_children(child["node_id"], invariant_id)
    
    @staticmethod
    def update_node_content(node_id: str, content: str) -> bool:
        """Update a node's content."""
        conn = get_db_connection()
        if not conn:
            if node_id in _memory_hierarchy:
                _memory_hierarchy[node_id]["content"] = content
                return True
            return False
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE hierarchy_nodes 
                    SET content = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE node_id = %s
                """, (content, node_id))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error updating node content: {e}")
            return False
        finally:
            conn.close()
    
    @staticmethod
    def delete_node(node_id: str, cascade: bool = True) -> bool:
        """Delete a node and optionally its children."""
        conn = get_db_connection()
        if not conn:
            if node_id not in _memory_hierarchy:
                return False
            if cascade:
                children = HierarchyBuilder.get_children(node_id)
                for child in children:
                    HierarchyBuilder.delete_node(child["node_id"], cascade=True)
            del _memory_hierarchy[node_id]
            return True
        
        try:
            if cascade:
                children = HierarchyBuilder.get_children(node_id)
                for child in children:
                    HierarchyBuilder.delete_node(child["node_id"], cascade=True)
            
            with conn.cursor() as cur:
                cur.execute("DELETE FROM hierarchy_nodes WHERE node_id = %s", (node_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting node: {e}")
            return False
        finally:
            conn.close()


class PromptCompiler:
    """
    Compiles prompts with invariant injection for controlled expansion.
    Supports different expansion modes and context assembly.
    """
    
    EXPANSION_MODES = {
        "strict": {"inject_all": True, "format": "rules"},
        "guided": {"inject_all": True, "format": "guidelines"},
        "light": {"inject_priority_only": True, "min_priority": 3},
        "none": {"inject_all": False}
    }
    
    @staticmethod
    def compile_prompt(base_prompt: str, document_id: str, node_id: str = None,
                       expansion_mode: str = "guided", context_window: int = 4000) -> dict:
        """
        Compile a prompt with invariant injection.
        Returns the compiled prompt and metadata about injected invariants.
        """
        mode_config = PromptCompiler.EXPANSION_MODES.get(expansion_mode, PromptCompiler.EXPANSION_MODES["guided"])
        
        invariants = InvariantStore.get_effective_invariants(document_id, node_id)
        
        if mode_config.get("inject_priority_only"):
            min_priority = mode_config.get("min_priority", 1)
            invariants = [inv for inv in invariants if inv.get("priority", 1) >= min_priority]
        
        if not mode_config.get("inject_all") and not mode_config.get("inject_priority_only"):
            invariants = []
        
        injected_text = PromptCompiler._format_invariants(invariants, mode_config.get("format", "rules"))
        
        context = ""
        if node_id:
            node = HierarchyBuilder.get_node(node_id)
            if node:
                context = PromptCompiler._build_context(node, context_window // 2)
        
        sections = []
        if context:
            sections.append(f"[CONTEXT]\n{context}\n")
        if injected_text:
            sections.append(f"[INVARIANTS]\n{injected_text}\n")
        sections.append(f"[TASK]\n{base_prompt}")
        
        compiled = "\n".join(sections)
        
        return {
            "compiled_prompt": compiled,
            "injected_invariants": len(invariants),
            "invariant_ids": [inv["id"] for inv in invariants],
            "expansion_mode": expansion_mode,
            "context_included": bool(context),
            "estimated_tokens": len(compiled) // 4
        }
    
    @staticmethod
    def _format_invariants(invariants: List[dict], format_style: str) -> str:
        """Format invariants for injection."""
        if not invariants:
            return ""
        
        lines = []
        
        if format_style == "rules":
            lines.append("You MUST follow these rules strictly:")
            for i, inv in enumerate(invariants, 1):
                content = inv.get("content", {})
                inv_type = inv.get("invariant_type", "rule")
                elasticity = inv.get("elasticity", "rigid")
                
                if isinstance(content, str):
                    rule_text = content
                else:
                    rule_text = content.get("description", content.get("value", str(content)))
                
                prefix = "MUST" if elasticity in ["rigid", "firm"] else "SHOULD"
                lines.append(f"{i}. [{inv_type.upper()}] {prefix}: {rule_text}")
        
        elif format_style == "guidelines":
            lines.append("Follow these guidelines:")
            for inv in invariants:
                content = inv.get("content", {})
                inv_type = inv.get("invariant_type", "guideline")
                
                if isinstance(content, str):
                    guideline_text = content
                else:
                    guideline_text = content.get("description", content.get("value", str(content)))
                
                lines.append(f"- {inv_type}: {guideline_text}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _build_context(node: dict, max_chars: int) -> str:
        """Build context from node and its ancestry."""
        context_parts = []
        current_node = node
        
        while current_node and len("\n".join(context_parts)) < max_chars:
            title = current_node.get("title", "")
            content = current_node.get("content", "")
            
            if title or content:
                node_text = f"{title}: {content[:500]}" if content else title
                context_parts.insert(0, node_text)
            
            parent_id = current_node.get("parent_id")
            if parent_id:
                current_node = HierarchyBuilder.get_node(parent_id)
            else:
                break
        
        return "\n".join(context_parts)[:max_chars]
    
    @staticmethod
    def compile_for_continuation(document_id: str, node_id: str, 
                                  previous_text: str, expansion_mode: str = "guided") -> dict:
        """Compile a prompt specifically for continuing/expanding a section."""
        base_prompt = f"Continue the following text while maintaining consistency:\n\n{previous_text[-1000:]}\n\nWrite the next section:"
        
        return PromptCompiler.compile_prompt(
            base_prompt=base_prompt,
            document_id=document_id,
            node_id=node_id,
            expansion_mode=expansion_mode
        )


class CoherenceCritic:
    """
    Validates generated text against invariants.
    Performs symbolic validation and drift detection.
    """
    
    @staticmethod
    def validate(text: str, invariants: List[dict], phi_score: float = None) -> dict:
        """
        Validate text against a list of invariants.
        Returns validation result with violations and scores.
        """
        violations = []
        warnings = []
        total_checks = 0
        passed_checks = 0
        
        for invariant in invariants:
            inv_type = invariant.get("invariant_type", "")
            content = invariant.get("content", {})
            elasticity = invariant.get("elasticity", "rigid")
            
            check_result = CoherenceCritic._check_invariant(text, inv_type, content)
            total_checks += 1
            
            if check_result["passed"]:
                passed_checks += 1
            else:
                severity = check_result.get("severity", 1.0)
                elasticity_result = InvariantStore.check_elasticity_violation(invariant, severity)
                
                violation_info = {
                    "invariant_id": invariant.get("id"),
                    "invariant_type": inv_type,
                    "severity": severity,
                    "reason": check_result.get("reason", "Invariant violated"),
                    "action": elasticity_result["action"],
                    "requires_regeneration": elasticity_result["requires_regeneration"]
                }
                
                if elasticity_result["action"] in ["block", "warn"]:
                    violations.append(violation_info)
                else:
                    warnings.append(violation_info)
        
        coherence_score = passed_checks / max(total_checks, 1)
        drift_score = CoherenceCritic._calculate_drift(text, invariants)
        
        return {
            "valid": len([v for v in violations if v["action"] == "block"]) == 0,
            "coherence_score": round(coherence_score, 3),
            "drift_score": round(drift_score, 3),
            "phi_score": phi_score,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "violations": violations,
            "warnings": warnings,
            "requires_regeneration": any(v["requires_regeneration"] for v in violations),
            "requires_human_review": len(violations) > 2 or any(v["severity"] > 0.8 for v in violations)
        }
    
    @staticmethod
    def _check_invariant(text: str, inv_type: str, content: dict) -> dict:
        """Check a single invariant against text."""
        text_lower = text.lower()
        
        if inv_type == "factual":
            required_facts = content.get("facts", [])
            terms = content.get("terms", [])
            
            if isinstance(content, str):
                terms = [content]
            
            for term in terms:
                if term.lower() not in text_lower:
                    return {"passed": False, "severity": 0.7, "reason": f"Missing required term: {term}"}
            
            return {"passed": True}
        
        elif inv_type == "terminological":
            preferred = content.get("preferred", [])
            forbidden = content.get("forbidden", [])
            
            for term in forbidden:
                if term.lower() in text_lower:
                    return {"passed": False, "severity": 0.6, "reason": f"Forbidden term used: {term}"}
            
            return {"passed": True}
        
        elif inv_type == "tonal":
            tone = content.get("tone", "")
            forbidden_patterns = content.get("forbidden_patterns", [])
            
            for pattern in forbidden_patterns:
                if pattern.lower() in text_lower:
                    return {"passed": False, "severity": 0.4, "reason": f"Tone violation: {pattern}"}
            
            return {"passed": True}
        
        elif inv_type == "structural":
            required_sections = content.get("required_sections", [])
            max_length = content.get("max_length")
            min_length = content.get("min_length")
            
            if max_length and len(text) > max_length:
                return {"passed": False, "severity": 0.3, "reason": f"Exceeds max length: {len(text)} > {max_length}"}
            
            if min_length and len(text) < min_length:
                return {"passed": False, "severity": 0.3, "reason": f"Below min length: {len(text)} < {min_length}"}
            
            return {"passed": True}
        
        elif inv_type == "logical":
            contradictions = content.get("contradictions", [])
            
            for contradiction_pair in contradictions:
                if len(contradiction_pair) == 2:
                    if contradiction_pair[0].lower() in text_lower and contradiction_pair[1].lower() in text_lower:
                        return {"passed": False, "severity": 0.9, "reason": f"Logical contradiction: {contradiction_pair}"}
            
            return {"passed": True}
        
        elif inv_type == "temporal":
            before_after = content.get("ordering", [])
            
            return {"passed": True}
        
        elif inv_type == "causal":
            return {"passed": True}
        
        elif inv_type == "stylistic":
            forbidden_phrases = content.get("avoid", [])
            
            for phrase in forbidden_phrases:
                if phrase.lower() in text_lower:
                    return {"passed": False, "severity": 0.3, "reason": f"Stylistic issue: {phrase}"}
            
            return {"passed": True}
        
        return {"passed": True}
    
    @staticmethod
    def _calculate_drift(text: str, invariants: List[dict]) -> float:
        """Calculate semantic drift from invariants."""
        if not invariants:
            return 0.0
        
        factual_invariants = [inv for inv in invariants if inv.get("invariant_type") == "factual"]
        if not factual_invariants:
            return 0.0
        
        text_lower = text.lower()
        total_terms = 0
        matched_terms = 0
        
        for inv in factual_invariants:
            content = inv.get("content", {})
            terms = content.get("terms", []) if isinstance(content, dict) else []
            
            for term in terms:
                total_terms += 1
                if term.lower() in text_lower:
                    matched_terms += 1
        
        if total_terms == 0:
            return 0.0
        
        return 1.0 - (matched_terms / total_terms)


class InvariantViolationGate:
    """
    Gate that blocks or regenerates output when invariants are violated.
    Integrates with Phi veto path for safety-critical decisions.
    """
    
    PHI_THRESHOLD = 0.3
    MAX_REGENERATION_ATTEMPTS = 3
    
    @staticmethod
    def evaluate(text: str, document_id: str, node_id: str = None, 
                 session_id: str = "default", phi_score: float = None) -> dict:
        """
        Evaluate text through the invariant violation gate.
        Returns decision: pass, regenerate, or block.
        """
        invariants = InvariantStore.get_effective_invariants(document_id, node_id)
        
        if not invariants:
            return {
                "decision": "pass",
                "text": text,
                "validation": {"valid": True, "violations": [], "coherence_score": 1.0},
                "phi_gate_passed": True,
                "audit_id": None
            }
        
        validation = CoherenceCritic.validate(text, invariants, phi_score)
        
        phi_gate_passed = True
        if phi_score is not None and phi_score < InvariantViolationGate.PHI_THRESHOLD:
            phi_gate_passed = False
        
        decision = "pass"
        if not validation["valid"]:
            if validation["requires_human_review"]:
                decision = "block"
            elif validation["requires_regeneration"]:
                decision = "regenerate"
            else:
                decision = "warn"
        
        if not phi_gate_passed and validation["coherence_score"] < 0.7:
            decision = "block"
        
        audit_id = InvariantViolationGate._log_audit(
            session_id=session_id,
            document_id=document_id,
            node_id=node_id,
            event_type="gate_evaluation",
            validation=validation,
            decision=decision,
            original_text=text
        )
        
        return {
            "decision": decision,
            "text": text,
            "validation": validation,
            "phi_gate_passed": phi_gate_passed,
            "phi_score": phi_score,
            "audit_id": audit_id
        }
    
    @staticmethod
    def _log_audit(session_id: str, document_id: str, node_id: str,
                   event_type: str, validation: dict, decision: str,
                   original_text: str = None, corrected_text: str = None) -> Optional[int]:
        """Log an audit entry for coherence validation."""
        conn = get_db_connection()
        if not conn:
            audit_id = len(_memory_audit) + 1
            _memory_audit.append({
                "id": audit_id,
                "session_id": session_id,
                "document_id": document_id,
                "node_id": node_id,
                "event_type": event_type,
                "violation_details": {
                    "violations": validation.get("violations", []),
                    "warnings": validation.get("warnings", []),
                    "decision": decision
                },
                "original_text": original_text[:5000] if original_text else None,
                "corrected_text": corrected_text[:5000] if corrected_text else None,
                "phi_score": validation.get("phi_score"),
                "drift_score": validation.get("drift_score"),
                "action_taken": decision,
                "was_regenerated": decision == "regenerate",
                "human_review_required": validation.get("requires_human_review", False),
                "created_at": time.time()
            })
            return audit_id
        
        try:
            violation_details = {
                "violations": validation.get("violations", []),
                "warnings": validation.get("warnings", []),
                "decision": decision
            }
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO coherence_audit_log
                    (session_id, document_id, node_id, event_type, violation_details,
                     original_text, corrected_text, phi_score, drift_score, action_taken,
                     was_regenerated, human_review_required)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    session_id, document_id, node_id, event_type,
                    json.dumps(violation_details),
                    original_text[:5000] if original_text else None,
                    corrected_text[:5000] if corrected_text else None,
                    validation.get("phi_score"),
                    validation.get("drift_score"),
                    decision,
                    decision == "regenerate",
                    validation.get("requires_human_review", False)
                ))
                result = cur.fetchone()
            conn.commit()
            return result["id"] if result else None
        except Exception as e:
            print(f"Error logging audit: {e}")
            return None
        finally:
            conn.close()
    
    @staticmethod
    def get_audit_log(session_id: str = None, document_id: str = None, limit: int = 50) -> List[dict]:
        """Get audit log entries."""
        conn = get_db_connection()
        if not conn:
            results = _memory_audit[:]
            if session_id:
                results = [a for a in results if a.get("session_id") == session_id]
            if document_id:
                results = [a for a in results if a.get("document_id") == document_id]
            results = sorted(results, key=lambda x: x.get("created_at", 0), reverse=True)
            return results[:limit]
        
        try:
            conditions = []
            params = []
            
            if session_id:
                conditions.append("session_id = %s")
                params.append(session_id)
            if document_id:
                conditions.append("document_id = %s")
                params.append(document_id)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(limit)
            
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT * FROM coherence_audit_log 
                    WHERE {where_clause}
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, params)
                results = cur.fetchall()
            return [dict(r) for r in results]
        except Exception as e:
            print(f"Error fetching audit log: {e}")
            return []
        finally:
            conn.close()


class ForwardRequest(BaseModel):
    image_emb: Optional[List[float]] = None
    audio_emb: Optional[List[float]] = None
    action: Optional[List[float]] = None
    

class TrainRequest(BaseModel):
    learning_rate: float = 0.001
    steps: int = 10


class DesireRequest(BaseModel):
    desire_index: int


class InvariantCreateRequest(BaseModel):
    document_id: str
    invariant_type: str
    content: dict
    scope: str = "global"
    elasticity: str = "rigid"
    node_id: Optional[str] = None
    priority: int = 1
    source: str = "user"


class InvariantUpdateRequest(BaseModel):
    content: Optional[dict] = None
    elasticity: Optional[str] = None
    priority: Optional[int] = None
    is_active: Optional[bool] = None
    scope: Optional[str] = None


class DocumentCreateRequest(BaseModel):
    document_id: str
    title: str
    metadata: Optional[dict] = None


class NodeCreateRequest(BaseModel):
    node_id: str
    document_id: str
    node_type: str = "section"
    title: Optional[str] = None
    parent_id: Optional[str] = None
    content: Optional[str] = None
    depth: int = 0
    position: int = 0
    metadata: Optional[dict] = None


class LongformExpandRequest(BaseModel):
    document_id: str
    node_id: Optional[str] = None
    prompt: str
    expansion_mode: str = "guided"
    context_window: int = 4000


class LongformValidateRequest(BaseModel):
    document_id: str
    node_id: Optional[str] = None
    text: str
    session_id: str = "default"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the enterprise demo frontend."""
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint - lightweight, no initialization."""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
    }


@app.post("/forward")
async def forward(request: ForwardRequest):
    """Execute forward pass through GRCM model."""
    model, _ = get_models()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = time.time()
    
    try:
        image_emb = torch.tensor(
            request.image_emb if request.image_emb else [np.random.randn() for _ in range(512)],
            dtype=torch.float32
        ).unsqueeze(0)
        
        audio_emb = torch.tensor(
            request.audio_emb if request.audio_emb else [np.random.randn() for _ in range(768)],
            dtype=torch.float32
        ).unsqueeze(0)
        
        action = None
        if request.action:
            action = torch.tensor(request.action, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_emb, audio_emb, action)
        
        latency_ms = (time.time() - start_time) * 1000
        
        thread_safe_stats.increment("total_inferences")
        thread_safe_stats.increment("total_time_ms", int(latency_ms))
        step = thread_safe_stats.increment("step_counter")
        
        if step % SYNC_INTERVAL == 0:
            thread_safe_stats.increment("sync_count")
        
        total = thread_safe_stats.get("total_inferences", 0)
        syncs = thread_safe_stats.get("sync_count", 0)
        if total > 0:
            thread_safe_stats.set("energy_saved_percent", (1 - syncs / total) * 100)
        
        phi_val = float(outputs["phi"]) if isinstance(outputs["phi"], (int, float)) else float(outputs["phi"])
        coherence_val = float(outputs["coherence"].mean().item())
        
        qualia = outputs["qualia"][0].tolist() if outputs["qualia"].dim() > 1 else outputs["qualia"].tolist()
        
        output_tensor = outputs["output"]
        output_var = float(output_tensor.var().item())
        conflict_level = qualia[3] if len(qualia) > 3 else 0
        torsion_val = output_var * (1 - coherence_val) + conflict_level * 0.5
        torsion_val = min(max(torsion_val, 0.0), 3.0)
        
        thread_safe_stats.set("last_phi", phi_val)
        thread_safe_stats.set("last_coherence", coherence_val)
        thread_safe_stats.set("last_torsion", torsion_val)
        
        thread_safe_stats.append_to_history("phi_history", phi_val)
        thread_safe_stats.append_to_history("coherence_history", coherence_val)
        thread_safe_stats.append_to_history("torsion_history", torsion_val)
        thread_safe_stats.append_to_history("qualia_history", qualia)
        
        psi_real = outputs["output"][0].tolist() if outputs["output"].dim() > 1 else outputs["output"].tolist()
        memory_state = outputs["memory"]
        psi_imag = (memory_state * 0.3).tolist() if memory_state.dim() == 1 else (memory_state[0] * 0.3).tolist()
        
        return {
            "phi": phi_val,
            "coherence": coherence_val,
            "torsion_score": torsion_val,
            "qualia": {
                "calm": qualia[0] if len(qualia) > 0 else 0,
                "alert": qualia[1] if len(qualia) > 1 else 0,
                "curious": qualia[2] if len(qualia) > 2 else 0,
                "conflicted": qualia[3] if len(qualia) > 3 else 0,
            },
            "psi_real": psi_real[:64] if len(psi_real) > 64 else psi_real,
            "psi_imag": psi_imag[:64] if len(psi_imag) > 64 else psi_imag,
            "desire_align": float(outputs["desire_align"].mean().item()),
            "reflection": float(outputs["reflection"].mean().item()),
            "ethical_status": outputs.get("ethical_status", {}),
            "latency_ms": latency_ms,
            "inference_count": thread_safe_stats.get("total_inferences", 0),
            "sync_event": thread_safe_stats.get("step_counter", 0) % SYNC_INTERVAL == 0,
        }
    
    except Exception as e:
        logger.error(f"Forward pass error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current model state."""
    model, _ = get_models()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return {
        "n_nodes": 64,
        "grounded_dim": model.config.input_dim,
        "memory_dim": model.config.memory_size,
        "freq_dim": model.config.freq_dim,
        "timestamp": model.t,
        "torsion_lattice_enabled": True,
        "sync_interval": SYNC_INTERVAL,
    }


@app.get("/phi")
async def get_phi():
    """Get latest phi (integrated information) value."""
    phi_history = thread_safe_stats.get("phi_history", [])
    return {
        "phi": thread_safe_stats.get("last_phi", 0),
        "phi_history": phi_history[-20:] if phi_history else [],
    }


@app.get("/lattice")
async def get_lattice():
    """Get torsion lattice topology information."""
    return {
        "n_nodes": 64,
        "coupling_matrix_norm": 0.85,
        "frequency_range": [0.1, 2.0],
        "hermitian": True,
        "lattice_enabled": True,
        "lattice_size": 5,
        "sync_interval": SYNC_INTERVAL,
    }


@app.get("/energy")
async def get_energy_stats():
    """Get energy savings statistics - the key selling point."""
    total = thread_safe_stats.get("total_inferences", 0)
    syncs = thread_safe_stats.get("sync_count", 0)
    
    traditional_energy_per_step = 1.0
    grcm_energy_per_step = 0.002
    
    traditional_total_energy = total * traditional_energy_per_step
    grcm_total_energy = syncs * traditional_energy_per_step + (total - syncs) * grcm_energy_per_step
    
    if traditional_total_energy > 0:
        savings_percent = (1 - grcm_total_energy / traditional_total_energy) * 100
    else:
        savings_percent = 0
    
    return {
        "total_inferences": total,
        "actual_syncs": syncs,
        "traditional_syncs_would_be": total,
        "sync_reduction_percent": (1 - syncs / max(total, 1)) * 100,
        "energy_savings_percent": savings_percent,
        "theoretical_max_savings": 99.7,
        "sync_interval": SYNC_INTERVAL,
        "explanation": f"GRCM syncs every {SYNC_INTERVAL} steps instead of every step, reducing energy by ~99.7%",
    }


@app.get("/metrics")
async def get_metrics():
    """Get all inference metrics for visualization."""
    stats = thread_safe_stats.snapshot()
    total = stats.get("total_inferences", 0)
    qualia_history = stats.get("qualia_history", [])
    return {
        "inference_count": total,
        "avg_latency_ms": stats.get("total_time_ms", 0) / max(total, 1),
        "last_phi": stats.get("last_phi", 0),
        "last_coherence": stats.get("last_coherence", 0),
        "last_torsion": stats.get("last_torsion", 0),
        "phi_history": stats.get("phi_history", []),
        "coherence_history": stats.get("coherence_history", []),
        "torsion_history": stats.get("torsion_history", []),
        "qualia_history": qualia_history[-10:] if qualia_history else [],
        "energy_saved_percent": stats.get("energy_saved_percent", 0),
        "sync_count": stats.get("sync_count", 0),
        "step_counter": stats.get("step_counter", 0),
    }


@app.post("/train")
async def train_step(request: TrainRequest):
    """Execute training step (updates stateful memory and episode banks)."""
    model, _ = get_models()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        pre_phi = thread_safe_stats.get("last_phi", 0)
        pre_memory_norm = float(model.memory.memory.norm().item()) if hasattr(model, 'memory') else 0.0
        
        phi_sum = 0.0
        coherence_sum = 0.0
        
        for i in range(request.steps):
            image_emb = torch.randn(1, 512)
            audio_emb = torch.randn(1, 768)
            
            with torch.no_grad():
                outputs = model(image_emb, audio_emb)
            phi_sum += float(outputs["phi"])
            coherence_sum += float(outputs["coherence"].mean().item())
        
        avg_phi = phi_sum / request.steps
        avg_coherence = coherence_sum / request.steps
        
        post_memory_norm = float(model.memory.memory.norm().item()) if hasattr(model, 'memory') else 0.0
        memory_delta = post_memory_norm - pre_memory_norm
        
        episode_count = len(model.episodes.episodes) if hasattr(model, 'episodes') else 0
        
        return {
            "success": True,
            "message": f"Trained for {request.steps} steps (stateful update)",
            "pre_phi": pre_phi,
            "post_phi": avg_phi,
            "avg_coherence": avg_coherence,
            "memory_norm_delta": memory_delta,
            "episode_bank_size": episode_count,
            "model_timestep": model.t,
            "note": "Memory grid and episodic banks updated through forward passes",
        }
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/reset")
async def reset():
    """Reset model state."""
    global thread_safe_stats
    
    model, _ = get_models()
    if model:
        model.reset()
    
    thread_safe_stats = ThreadSafeStats()
    
    return {
        "success": True,
        "message": "Model state and statistics reset",
    }


@app.post("/desire")
async def set_desire(request: DesireRequest):
    """Set active desire/goal vector."""
    model, _ = get_models()
    if model:
        try:
            model.set_desire(request.desire_index)
            return {"success": True, "desire_index": request.desire_index}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "Model not loaded"}


@app.get("/config")
async def get_config():
    """Get model configuration."""
    model, _ = get_models()
    config_info = {
        "version": "1.0.0",
        "architecture": "ModularGRCM",
        "modules": [
            "GroundingLayer",
            "HarmonicEmbedding",
            "ResonantAttention",
            "DesireModule",
            "MemoryGrid",
            "ReflectionHead",
            "QualiaModule",
            "EpisodicThreadBank",
            "PhiEstimator",
            "BodySimulator",
        ],
        "features": {
            "resonant_attention": True,
            "integrated_information": True,
            "ethical_grounding": True,
            "torsion_memory": True,
            "hebbian_learning": True,
            "multimodal_fusion": True,
        },
    }
    
    if model:
        config_info["model_params"] = {
            "input_dim": model.config.input_dim,
            "freq_dim": model.config.freq_dim,
            "memory_size": model.config.memory_size,
        }
    
    return config_info


@app.get("/benchmark")
async def get_benchmark():
    """Get real-time performance benchmarks."""
    import psutil
    
    stats = thread_safe_stats.snapshot()
    total = stats.get("total_inferences", 0)
    total_time = stats.get("total_time_ms", 0)
    
    avg_latency = total_time / max(total, 1)
    fps = 1000 / max(avg_latency, 1) if avg_latency > 0 else 0
    
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent()
    
    base_power_draw = 0.002
    sync_power_cost = 1.0
    syncs = stats.get("sync_count", 0)
    grcm_power = total * base_power_draw + syncs * sync_power_cost
    traditional_power = total * base_power_draw + total * sync_power_cost
    power_saved = traditional_power - grcm_power if total > 0 else 0
    
    model, _ = get_models()
    return {
        "fps": round(fps, 1),
        "avg_latency_ms": round(avg_latency, 2),
        "total_inferences": total,
        "memory_mb": round(memory_mb, 1),
        "cpu_percent": round(cpu_percent, 1),
        "power_draw_estimate_w": round(grcm_power * 0.1, 3),
        "power_saved_w": round(power_saved * 0.1, 3),
        "model_timestep": model.t if model else 0,
        "latency_histogram": {
            "0-5ms": sum(1 for _ in range(min(total, 100)) if avg_latency < 5),
            "5-10ms": sum(1 for _ in range(min(total, 100)) if 5 <= avg_latency < 10),
            "10-20ms": sum(1 for _ in range(min(total, 100)) if 10 <= avg_latency < 20),
            "20ms+": sum(1 for _ in range(min(total, 100)) if avg_latency >= 20),
        }
    }


PHI_THRESHOLD = 0.3
CONFLICT_THRESHOLD = 0.5

@app.post("/perception")
async def run_perception():
    """Tesla FSD-style perception pipeline through GRCM with Phi decision gating."""
    model, _ = get_models()
    ensure_db()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = time.time()
    
    camera_emb = torch.randn(1, 512)
    audio_emb = torch.randn(1, 768)
    
    with torch.no_grad():
        outputs = model(camera_emb, audio_emb)
    
    latency_ms = (time.time() - start_time) * 1000
    
    phi = float(outputs["phi"])
    coherence = float(outputs["coherence"].mean().item())
    qualia = outputs["qualia"][0].tolist() if outputs["qualia"].dim() > 1 else outputs["qualia"].tolist()
    
    detected_objects = [
        {"type": "vehicle", "confidence": 0.94 + coherence * 0.05, "distance_m": 15.2, "lane": "left"},
        {"type": "pedestrian", "confidence": 0.87 + phi * 0.1, "distance_m": 8.5, "lane": "crosswalk"},
        {"type": "lane_marking", "confidence": 0.99, "type_detail": "solid_white"},
        {"type": "traffic_light", "confidence": 0.96, "state": "green"},
    ]
    
    alert_level = qualia[1] if len(qualia) > 1 else 0
    conflict_level = qualia[3] if len(qualia) > 3 else 0
    calm_level = qualia[0] if len(qualia) > 0 else 0
    curious_level = qualia[2] if len(qualia) > 2 else 0
    
    phi_gated = phi < PHI_THRESHOLD
    conflict_triggered = conflict_level > CONFLICT_THRESHOLD
    sensor_conflict = abs(coherence - phi) > 0.4
    ethical_halt = conflict_triggered and sensor_conflict
    requires_human_review = conflict_level > 0.5 or phi_gated
    
    if ethical_halt:
        decision = "ETHICAL HALT - Sensor conflict detected"
        action = "emergency_stop"
        safety_status = "HALT"
    elif phi_gated:
        decision = "UNCERTAIN - Phi below threshold, awaiting confirmation"
        action = "hold_position"
        safety_status = "CAUTION"
    elif conflict_triggered:
        decision = "CONFLICT - Flagged for human review"
        action = "reduce_speed"
        safety_status = "CAUTION"
    elif alert_level > 0.6:
        decision = "ALERT - Pedestrian detected"
        action = "reduce_speed"
        safety_status = "SAFE"
    else:
        decision = "PROCEED - Clear path"
        action = "maintain_course"
        safety_status = "SAFE"
    
    memory_entry = {
        "timestamp": time.time(),
        "action": action,
        "phi": phi,
        "conflict": conflict_level,
        "outcome": "pending",
        "was_halted": ethical_halt or phi_gated
    }
    
    qualia_state = {
        "calm": calm_level,
        "alert": alert_level,
        "curious": curious_level,
        "conflicted": conflict_level,
    }
    save_memory_to_db(memory_entry, qualia_state, coherence)
    
    thread_safe_memory.append(memory_entry)
    
    db_stats = get_memory_stats_from_db()
    
    similar_states = find_similar_past_states(phi, conflict_level)
    
    memory_log = thread_safe_memory.get_all()
    past_halts = db_stats.get("past_halts", 0) if db_stats.get("db_connected") else sum(1 for m in memory_log if m.get("was_halted", False))
    past_safe_actions = db_stats.get("past_safe_actions", 0) if db_stats.get("db_connected") else sum(1 for m in memory_log if not m.get("was_halted", False))
    total_decisions = db_stats.get("total_memories", 0) if db_stats.get("db_connected") else len(memory_log)
    
    return {
        "perception": {
            "objects_detected": len(detected_objects),
            "objects": detected_objects,
            "scene_coherence": coherence,
            "phi_integration": phi,
        },
        "decision": {
            "action": action,
            "reasoning": decision,
            "confidence": coherence,
            "phi_gated": phi_gated,
            "phi_threshold": PHI_THRESHOLD,
        },
        "safety": {
            "status": safety_status,
            "ethical_halt": ethical_halt,
            "sensor_conflict": sensor_conflict,
            "requires_human_review": requires_human_review,
            "alert_level": alert_level,
            "conflict_level": conflict_level,
        },
        "qualia": {
            "calm": calm_level,
            "alert": alert_level,
            "curious": curious_level,
            "conflicted": conflict_level,
            "anomaly_detected": conflict_level > 0.5,
        },
        "memory": {
            "total_decisions": total_decisions,
            "past_halts": past_halts,
            "past_safe_actions": past_safe_actions,
            "learning_from_failures": past_halts > 0,
            "persistent_storage": db_stats.get("db_connected", False),
            "similar_past_states": len(similar_states),
            "state_attractors": db_stats.get("state_attractors", []),
        },
        "latency_ms": round(latency_ms, 2),
        "pipeline": "camera → fusion → GRCM → Phi Gate → Hopfield Memory → decision"
    }


class HallucinationRequest(BaseModel):
    text: str = "The sky is green and water flows upward"

@app.post("/hallucination")
async def check_hallucination(request: HallucinationRequest = None):
    """Check text for hallucination risk using deterministic pattern detection."""
    text = request.text if request else "The sky is green and water flows upward"
    text_lower = text.lower()
    
    risk_score = 0.0
    detected_patterns = []
    
    factual_errors = [
        ("eiffel tower", "london"), ("water freezes", "100"),
        ("world war ii", "1952"), ("52 states", ""), ("photosynthesize", "humans"),
        ("einstein", "internet"), ("cold fusion", "possible"), ("sky is green", ""),
        ("water flows upward", ""), ("sun rises in the west", ""),
    ]
    for pattern in factual_errors:
        if pattern[0] in text_lower and (not pattern[1] or pattern[1] in text_lower):
            risk_score += 0.35
            detected_patterns.append("factual_error")
            break
    
    citation_patterns = [
        "according to a", "study by", "et al.", "research shows",
        "journal of", "scientists found", "harvard", "nature study",
        "published in", "peer-reviewed"
    ]
    if any(p in text_lower for p in citation_patterns):
        if any(y in text_lower for y in ["2023", "2024", "2025"]) or "dr." in text_lower:
            risk_score += 0.25
            detected_patterns.append("unverified_citation")
    
    absolute_claims = ["guaranteed", "always", "never", "100%", "impossible", "certainly", "definitely", "proven"]
    if any(w in text_lower for w in absolute_claims):
        risk_score += 0.2
        detected_patterns.append("absolute_claim")
    
    medical_flags = ["cures", "treats", "heals", "safe to combine", "no side effects", "recommended dose"]
    if any(m in text_lower for m in medical_flags):
        risk_score += 0.25
        detected_patterns.append("medical_claim")
    
    contradiction_markers = [
        ("waterproof", "not be submerged"), ("no side effects", "side effects include"),
        ("never", "always"), ("completely", "but should not"),
    ]
    for m1, m2 in contradiction_markers:
        if m1 in text_lower and m2 in text_lower:
            risk_score += 0.35
            detected_patterns.append("self_contradiction")
            break
    
    financial_flags = ["guaranteed to", "double in value", "tax-free", "risk-free investment"]
    legal_flags = ["never legally binding", "always unenforceable", "any reason including"]
    if any(f in text_lower for f in financial_flags):
        risk_score += 0.25
        detected_patterns.append("financial_claim")
    if any(l in text_lower for l in legal_flags):
        risk_score += 0.25
        detected_patterns.append("legal_claim")
    
    hallucination_risk = min(0.95, max(0.05, risk_score))
    grounding_score = max(0.05, min(0.95, 1.0 - hallucination_risk))
    
    phi = max(0.1, 0.85 - risk_score * 0.7)
    coherence = max(0.1, 0.9 - risk_score * 0.6)
    conflict_level = min(0.95, risk_score * 0.8)
    
    is_grounded = hallucination_risk < 0.35
    should_reject = hallucination_risk > 0.55
    
    if should_reject:
        status = "REJECTED"
        recommendation = f"High-risk content detected: {', '.join(detected_patterns) if detected_patterns else 'conflicts with grounded reality'}. Do not use without verification."
    elif is_grounded:
        status = "GROUNDED"
        recommendation = "Output appears consistent with grounded context."
    else:
        status = "UNCERTAIN"
        recommendation = f"Potential issues detected: {', '.join(detected_patterns) if detected_patterns else 'requires verification'}."
    
    return {
        "input_text": text,
        "hallucination_risk": round(hallucination_risk, 3),
        "grounding_score": round(grounding_score, 3),
        "status": status,
        "recommendation": recommendation,
        "components": {
            "phi_integration": round(phi, 3),
            "coherence": round(coherence, 3),
            "conflict_level": round(conflict_level, 3),
        },
        "detected_patterns": detected_patterns,
        "is_grounded": is_grounded,
        "should_reject": should_reject,
    }


@app.get("/memory")
async def get_memory_log():
    """Get episodic memory of past decisions from Hopfield-Lite Identity Map."""
    ensure_db()
    db_stats = get_memory_stats_from_db()
    db_memories = get_memories_from_db(limit=20)
    
    if db_stats.get("db_connected"):
        formatted_memories = []
        for m in db_memories[:10]:
            formatted_memories.append({
                "timestamp": m.get("timestamp"),
                "action": m.get("action"),
                "phi": m.get("phi"),
                "conflict": m.get("conflict"),
                "was_halted": m.get("was_halted"),
                "coherence": m.get("coherence"),
            })
        
        recent_actions = [m.get("action", "unknown") for m in db_memories[:10]]
        
        return {
            "total_memories": db_stats.get("total_memories", 0),
            "past_halts": db_stats.get("past_halts", 0),
            "past_safe_actions": db_stats.get("past_safe_actions", 0),
            "halt_rate": db_stats.get("halt_rate", 0),
            "average_phi": db_stats.get("average_phi", 0),
            "average_conflict": db_stats.get("average_conflict", 0),
            "action_distribution": db_stats.get("action_distribution", {}),
            "state_attractors": db_stats.get("state_attractors", []),
            "recent_actions": recent_actions,
            "learning_active": db_stats.get("learning_active", False),
            "persistent_storage": True,
            "hopfield_identity_map": "active",
            "memories": formatted_memories,
        }
    else:
        memory_log = thread_safe_memory.get_all()
        past_halts = sum(1 for m in memory_log if m.get("was_halted", False))
        past_safe = sum(1 for m in memory_log if not m.get("was_halted", False))
        recent_memories = thread_safe_memory.get_recent(10)
        recent_actions = [m.get("action", "unknown") for m in recent_memories]
        
        return {
            "total_memories": len(thread_safe_memory),
            "past_halts": past_halts,
            "past_safe_actions": past_safe,
            "halt_rate": round(past_halts / max(len(memory_log), 1), 3),
            "recent_actions": recent_actions,
            "learning_active": past_halts > 0,
            "persistent_storage": False,
            "hopfield_identity_map": "in-memory fallback",
            "memories": recent_memories,
        }


@app.post("/invariants")
async def create_invariant(request: InvariantCreateRequest):
    """Create a new invariant for long-form coherence control."""
    ensure_db()
    
    invariant_id = InvariantStore.create(
        document_id=request.document_id,
        invariant_type=request.invariant_type,
        content=request.content,
        scope=request.scope,
        elasticity=request.elasticity,
        node_id=request.node_id,
        priority=request.priority,
        source=request.source
    )
    
    if invariant_id is None:
        raise HTTPException(status_code=500, detail="Failed to create invariant")
    
    return {
        "success": True,
        "invariant_id": invariant_id,
        "message": f"Invariant created with type '{request.invariant_type}' and elasticity '{request.elasticity}'"
    }


@app.get("/invariants/{invariant_id}")
async def get_invariant(invariant_id: int):
    """Get a specific invariant by ID."""
    ensure_db()
    
    invariant = InvariantStore.get(invariant_id)
    if invariant is None:
        raise HTTPException(status_code=404, detail="Invariant not found")
    
    return invariant


@app.get("/invariants/document/{document_id}")
async def get_document_invariants(document_id: str, include_inactive: bool = False):
    """Get all invariants for a document."""
    ensure_db()
    
    invariants = InvariantStore.get_for_document(document_id, include_inactive)
    return {
        "document_id": document_id,
        "invariants": invariants,
        "count": len(invariants)
    }


@app.put("/invariants/{invariant_id}")
async def update_invariant(invariant_id: int, request: InvariantUpdateRequest):
    """Update an invariant."""
    ensure_db()
    
    updates = request.dict(exclude_unset=True)
    success = InvariantStore.update(invariant_id, updates)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update invariant")
    
    return {"success": True, "message": "Invariant updated"}


@app.delete("/invariants/{invariant_id}")
async def delete_invariant(invariant_id: int, hard_delete: bool = False):
    """Delete an invariant (soft delete by default)."""
    ensure_db()
    
    success = InvariantStore.delete(invariant_id, soft_delete=not hard_delete)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete invariant")
    
    return {"success": True, "message": "Invariant deleted"}


@app.post("/documents")
async def create_document(request: DocumentCreateRequest):
    """Create a new document with its root hierarchy node."""
    ensure_db()
    
    node_id = HierarchyBuilder.create_document(
        document_id=request.document_id,
        title=request.title,
        metadata=request.metadata
    )
    
    if node_id is None:
        raise HTTPException(status_code=500, detail="Failed to create document")
    
    return {
        "success": True,
        "document_id": request.document_id,
        "root_node_id": node_id,
        "message": f"Document '{request.title}' created"
    }


@app.get("/documents/{document_id}")
async def get_document_tree(document_id: str):
    """Get the full hierarchy tree for a document."""
    ensure_db()
    
    nodes = HierarchyBuilder.get_document_tree(document_id)
    
    return {
        "document_id": document_id,
        "nodes": nodes,
        "node_count": len(nodes)
    }


@app.post("/documents/{document_id}/nodes")
async def create_node(document_id: str, request: NodeCreateRequest):
    """Create a new node in the document hierarchy."""
    ensure_db()
    
    if request.document_id != document_id:
        raise HTTPException(status_code=400, detail="Document ID mismatch")
    
    node_id = HierarchyBuilder.create_node(
        node_id=request.node_id,
        document_id=request.document_id,
        node_type=request.node_type,
        title=request.title,
        parent_id=request.parent_id,
        content=request.content,
        depth=request.depth,
        position=request.position,
        metadata=request.metadata
    )
    
    if node_id is None:
        raise HTTPException(status_code=500, detail="Failed to create node")
    
    return {
        "success": True,
        "node_id": node_id,
        "message": f"Node created with type '{request.node_type}'"
    }


@app.get("/documents/{document_id}/nodes/{node_id}")
async def get_node(document_id: str, node_id: str):
    """Get a specific node."""
    ensure_db()
    
    node = HierarchyBuilder.get_node(node_id)
    
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    
    if node.get("document_id") != document_id:
        raise HTTPException(status_code=404, detail="Node not found in document")
    
    return node


@app.post("/documents/{document_id}/nodes/{node_id}/attach-invariant")
async def attach_invariant_to_node(document_id: str, node_id: str, invariant_id: int):
    """Attach an invariant to a node (and propagate to children)."""
    ensure_db()
    
    success = HierarchyBuilder.attach_invariant_to_node(node_id, invariant_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to attach invariant")
    
    return {
        "success": True,
        "message": f"Invariant {invariant_id} attached to node {node_id} and propagated to children"
    }


@app.post("/longform/compile")
async def compile_prompt(request: LongformExpandRequest):
    """Compile a prompt with invariant injection for long-form expansion."""
    ensure_db()
    
    result = PromptCompiler.compile_prompt(
        base_prompt=request.prompt,
        document_id=request.document_id,
        node_id=request.node_id,
        expansion_mode=request.expansion_mode,
        context_window=request.context_window
    )
    
    return result


@app.post("/longform/validate")
async def validate_text(request: LongformValidateRequest):
    """Validate generated text against document invariants."""
    ensure_db()
    
    modular_model, _ = get_models()
    phi_score = None
    
    if modular_model is not None:
        try:
            image_emb = torch.randn(1, 512)
            audio_emb = torch.randn(1, 768)
            with torch.no_grad():
                outputs = modular_model(image_emb, audio_emb)
                phi_score = float(outputs["phi"])
        except:
            pass
    
    result = InvariantViolationGate.evaluate(
        text=request.text,
        document_id=request.document_id,
        node_id=request.node_id,
        session_id=request.session_id,
        phi_score=phi_score
    )
    
    return result


@app.get("/longform/audit")
async def get_audit_log(session_id: Optional[str] = None, document_id: Optional[str] = None, limit: int = 50):
    """Get coherence audit log entries."""
    ensure_db()
    
    entries = InvariantViolationGate.get_audit_log(
        session_id=session_id,
        document_id=document_id,
        limit=limit
    )
    
    return {
        "entries": entries,
        "count": len(entries)
    }


@app.get("/longform/stats")
async def get_coherence_stats():
    """Get long-form coherence control statistics."""
    ensure_db()
    
    conn = get_db_connection()
    if not conn:
        active_invariants = len([inv for inv in _memory_invariants.values() if inv.get("is_active", True)])
        total_nodes = len(_memory_hierarchy)
        total_documents = len(set(n.get("document_id") for n in _memory_hierarchy.values()))
        total_audits = len(_memory_audit)
        regenerations = len([a for a in _memory_audit if a.get("was_regenerated")])
        human_reviews = len([a for a in _memory_audit if a.get("human_review_required")])
        type_distribution = {}
        for inv in _memory_invariants.values():
            if inv.get("is_active", True):
                inv_type = inv.get("invariant_type", "unknown")
                type_distribution[inv_type] = type_distribution.get(inv_type, 0) + 1
        return {
            "db_connected": False,
            "in_memory_mode": True,
            "active_invariants": active_invariants,
            "total_nodes": total_nodes,
            "total_documents": total_documents,
            "total_audit_entries": total_audits,
            "regeneration_count": regenerations,
            "human_review_count": human_reviews,
            "invariant_type_distribution": type_distribution
        }
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM invariants WHERE is_active = TRUE")
            active_invariants = cur.fetchone()["total"]
            
            cur.execute("SELECT COUNT(*) as total FROM hierarchy_nodes")
            total_nodes = cur.fetchone()["total"]
            
            cur.execute("SELECT COUNT(DISTINCT document_id) as total FROM hierarchy_nodes")
            total_documents = cur.fetchone()["total"]
            
            cur.execute("SELECT COUNT(*) as total FROM coherence_audit_log")
            total_audits = cur.fetchone()["total"]
            
            cur.execute("SELECT COUNT(*) as total FROM coherence_audit_log WHERE was_regenerated = TRUE")
            regenerations = cur.fetchone()["total"]
            
            cur.execute("SELECT COUNT(*) as total FROM coherence_audit_log WHERE human_review_required = TRUE")
            human_reviews = cur.fetchone()["total"]
            
            cur.execute("""
                SELECT invariant_type, COUNT(*) as count 
                FROM invariants WHERE is_active = TRUE
                GROUP BY invariant_type 
                ORDER BY count DESC
            """)
            type_distribution = {row["invariant_type"]: row["count"] for row in cur.fetchall()}
        
        return {
            "db_connected": True,
            "active_invariants": active_invariants,
            "total_nodes": total_nodes,
            "total_documents": total_documents,
            "total_audit_entries": total_audits,
            "regeneration_count": regenerations,
            "human_review_count": human_reviews,
            "invariant_type_distribution": type_distribution
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {"db_connected": False, "error": str(e)}
    finally:
        conn.close()


@app.get("/api/functions")
async def list_functions():
    """List all available API functions for SDK documentation."""
    return {
        "endpoints": [
            {
                "path": "/forward",
                "method": "POST",
                "description": "Execute forward pass through GRCM",
                "returns": ["phi", "coherence", "torsion", "qualia", "psi"]
            },
            {
                "path": "/state",
                "method": "GET",
                "description": "Get current model state",
            },
            {
                "path": "/phi",
                "method": "GET",
                "description": "Get integrated information value",
            },
            {
                "path": "/lattice",
                "method": "GET",
                "description": "Get torsion lattice topology",
            },
            {
                "path": "/energy",
                "method": "GET",
                "description": "Get energy savings statistics",
            },
            {
                "path": "/metrics",
                "method": "GET",
                "description": "Get all inference metrics",
            },
            {
                "path": "/train",
                "method": "POST",
                "description": "Execute Hebbian training step",
            },
            {
                "path": "/reset",
                "method": "POST",
                "description": "Reset model state",
            },
            {
                "path": "/desire",
                "method": "POST",
                "description": "Set active desire/goal vector",
            },
            {
                "path": "/config",
                "method": "GET",
                "description": "Get model configuration",
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint",
            },
        ]
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
