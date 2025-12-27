"""
PostgreSQL-backed Persistent Memory for GRCM
Implements Hopfield-Lite Identity Map for long-term memory continuity
"""
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """Single episodic memory entry"""
    id: int
    timestamp: datetime
    action: str
    phi: float
    coherence: float
    conflict: float
    qualia_state: Dict[str, float]
    was_halted: bool
    attractor: Optional[str]


@dataclass
class StateAttractor:
    """Learned state attractor pattern"""
    name: str
    phi_centroid: float
    coherence_centroid: float
    conflict_centroid: float
    occurrence_count: int
    last_seen: datetime


class PersistentMemory:
    """
    PostgreSQL-backed Hopfield-Lite Identity Map
    
    Features:
    - Episodic memory persists across restarts
    - State attractor learning (identifies repeating patterns)
    - Similar state retrieval for decision support
    - Full qualia/phi/coherence tracking
    
    Usage:
        memory = PersistentMemory(database_url)
        memory.save_decision(action, phi, coherence, conflict, qualia)
        
        similar = memory.find_similar_states(current_phi, current_conflict)
        for state in similar:
            print(f"Similar past: {state.action} (phi={state.phi})")
    """
    
    ATTRACTOR_THRESHOLDS = {
        'high_confidence': {'phi_min': 0.7, 'conflict_max': 0.3},
        'uncertain': {'phi_max': 0.4, 'conflict_max': 0.4},
        'conflicted': {'conflict_min': 0.5},
        'neutral': {}
    }
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize persistent memory
        
        Args:
            database_url: PostgreSQL connection string
                         Defaults to DATABASE_URL environment variable
        """
        self.database_url = database_url or os.environ.get('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL required for persistent memory")
            
        self._init_schema()
        
    def _get_connection(self):
        """Get database connection"""
        import psycopg2
        return psycopg2.connect(self.database_url)
        
    def _init_schema(self):
        """Initialize database schema"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action VARCHAR(255),
                phi FLOAT,
                coherence FLOAT,
                conflict FLOAT,
                qualia_calm FLOAT,
                qualia_alert FLOAT,
                qualia_curious FLOAT,
                qualia_conflicted FLOAT,
                was_halted BOOLEAN DEFAULT FALSE,
                attractor_name VARCHAR(64)
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS state_attractors (
                id SERIAL PRIMARY KEY,
                attractor_name VARCHAR(64) UNIQUE,
                phi_centroid FLOAT,
                coherence_centroid FLOAT,
                conflict_centroid FLOAT,
                occurrence_count INTEGER DEFAULT 1,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
    def save_decision(
        self,
        action: str,
        phi: float,
        coherence: float,
        conflict: float,
        qualia: Dict[str, float],
        was_halted: bool = False
    ) -> int:
        """
        Save a decision to episodic memory
        
        Args:
            action: Action taken or recommended
            phi: Integrated information value
            coherence: Coherence score
            conflict: Conflict level
            qualia: Qualia state dict (calm, alert, curious, conflicted)
            was_halted: Whether action was halted
            
        Returns:
            ID of saved memory entry
        """
        attractor = self._classify_attractor(phi, coherence, conflict)
        self._update_attractor(attractor, phi, coherence, conflict)
        
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO episodic_memory 
            (action, phi, coherence, conflict, qualia_calm, qualia_alert,
             qualia_curious, qualia_conflicted, was_halted, attractor_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            action, phi, coherence, conflict,
            qualia.get('calm', 0), qualia.get('alert', 0),
            qualia.get('curious', 0), qualia.get('conflicted', 0),
            was_halted, attractor
        ))
        
        memory_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return memory_id
        
    def _classify_attractor(self, phi: float, coherence: float, conflict: float) -> str:
        """Classify current state into attractor basin"""
        if phi >= 0.7 and conflict <= 0.3:
            return 'high_confidence'
        elif phi <= 0.4 and conflict <= 0.4:
            return 'uncertain'
        elif conflict >= 0.5:
            return 'conflicted'
        else:
            return 'neutral'
            
    def _update_attractor(self, name: str, phi: float, coherence: float, conflict: float):
        """Update attractor centroid with new observation"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO state_attractors (attractor_name, phi_centroid, coherence_centroid, 
                                          conflict_centroid, occurrence_count, last_seen)
            VALUES (%s, %s, %s, %s, 1, CURRENT_TIMESTAMP)
            ON CONFLICT (attractor_name) DO UPDATE SET
                phi_centroid = (state_attractors.phi_centroid * state_attractors.occurrence_count + %s) 
                             / (state_attractors.occurrence_count + 1),
                coherence_centroid = (state_attractors.coherence_centroid * state_attractors.occurrence_count + %s) 
                                   / (state_attractors.occurrence_count + 1),
                conflict_centroid = (state_attractors.conflict_centroid * state_attractors.occurrence_count + %s) 
                                  / (state_attractors.occurrence_count + 1),
                occurrence_count = state_attractors.occurrence_count + 1,
                last_seen = CURRENT_TIMESTAMP
        """, (name, phi, coherence, conflict, phi, coherence, conflict))
        
        conn.commit()
        cur.close()
        conn.close()
        
    def find_similar_states(
        self,
        phi: float,
        conflict: float,
        limit: int = 5,
        threshold: float = 0.2
    ) -> List[MemoryEntry]:
        """
        Find similar past states based on phi and conflict proximity
        
        Args:
            phi: Current phi value
            conflict: Current conflict level
            limit: Maximum number of results
            threshold: Maximum distance to consider similar
            
        Returns:
            List of similar past memory entries
        """
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, timestamp, action, phi, coherence, conflict,
                   qualia_calm, qualia_alert, qualia_curious, qualia_conflicted,
                   was_halted, attractor_name,
                   ABS(phi - %s) + ABS(conflict - %s) as distance
            FROM episodic_memory
            WHERE ABS(phi - %s) + ABS(conflict - %s) < %s
            ORDER BY distance ASC
            LIMIT %s
        """, (phi, conflict, phi, conflict, threshold, limit))
        
        results = []
        for row in cur.fetchall():
            results.append(MemoryEntry(
                id=row[0],
                timestamp=row[1],
                action=row[2],
                phi=row[3],
                coherence=row[4],
                conflict=row[5],
                qualia_state={
                    'calm': row[6],
                    'alert': row[7],
                    'curious': row[8],
                    'conflicted': row[9]
                },
                was_halted=row[10],
                attractor=row[11]
            ))
            
        cur.close()
        conn.close()
        
        return results
        
    def get_attractors(self) -> List[StateAttractor]:
        """Get all learned state attractors"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT attractor_name, phi_centroid, coherence_centroid, 
                   conflict_centroid, occurrence_count, last_seen
            FROM state_attractors
            ORDER BY occurrence_count DESC
        """)
        
        results = []
        for row in cur.fetchall():
            results.append(StateAttractor(
                name=row[0],
                phi_centroid=row[1],
                coherence_centroid=row[2],
                conflict_centroid=row[3],
                occurrence_count=row[4],
                last_seen=row[5]
            ))
            
        cur.close()
        conn.close()
        
        return results
        
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM episodic_memory")
        total = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM episodic_memory WHERE was_halted = TRUE")
        halted = cur.fetchone()[0]
        
        cur.execute("SELECT AVG(phi), AVG(conflict) FROM episodic_memory")
        row = cur.fetchone()
        avg_phi = row[0] or 0
        avg_conflict = row[1] or 0
        
        cur.close()
        conn.close()
        
        return {
            'total_memories': total,
            'halted_actions': halted,
            'halt_rate': halted / total if total > 0 else 0,
            'average_phi': avg_phi,
            'average_conflict': avg_conflict
        }
