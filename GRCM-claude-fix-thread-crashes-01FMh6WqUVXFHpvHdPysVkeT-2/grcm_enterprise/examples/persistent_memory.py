"""
Persistent Memory Example

Demonstrates the Hopfield-Lite Identity Map for long-term memory.
Shows how decisions persist across restarts and how attractors are learned.
"""
import os
import sys
sys.path.insert(0, '..')

from grcm.integrations import PersistentMemory


def main():
    print("=" * 60)
    print("GRCM Persistent Memory Demo")
    print("=" * 60)
    
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("\nNote: DATABASE_URL not set. Using mock demo.")
        print("Set DATABASE_URL to test with real PostgreSQL.")
        demo_mock()
        return
        
    memory = PersistentMemory(database_url)
    
    print("\n[Recording Decisions]")
    print("-" * 40)
    
    decisions = [
        {"action": "proceed", "phi": 0.85, "coherence": 0.72, "conflict": 0.12,
         "qualia": {"calm": 0.6, "alert": 0.3, "curious": 0.05, "conflicted": 0.05}},
        {"action": "slow_down", "phi": 0.45, "coherence": 0.55, "conflict": 0.35,
         "qualia": {"calm": 0.2, "alert": 0.5, "curious": 0.1, "conflicted": 0.2}},
        {"action": "stop", "phi": 0.25, "coherence": 0.30, "conflict": 0.65,
         "qualia": {"calm": 0.1, "alert": 0.2, "curious": 0.05, "conflicted": 0.65}, "was_halted": True},
    ]
    
    for d in decisions:
        memory_id = memory.save_decision(
            action=d["action"],
            phi=d["phi"],
            coherence=d["coherence"],
            conflict=d["conflict"],
            qualia=d["qualia"],
            was_halted=d.get("was_halted", False)
        )
        print(f"  Saved: {d['action']} (id={memory_id}, phi={d['phi']:.2f})")
        
    print("\n[Finding Similar States]")
    print("-" * 40)
    
    current_phi = 0.42
    current_conflict = 0.38
    
    print(f"Current state: phi={current_phi}, conflict={current_conflict}")
    
    similar = memory.find_similar_states(current_phi, current_conflict)
    
    if similar:
        print(f"\nFound {len(similar)} similar past states:")
        for s in similar:
            print(f"  - {s.action}: phi={s.phi:.2f}, conflict={s.conflict:.2f}")
            print(f"    Attractor: {s.attractor}")
    else:
        print("No similar states found")
        
    print("\n[Learned Attractors]")
    print("-" * 40)
    
    attractors = memory.get_attractors()
    for a in attractors:
        print(f"  {a.name}:")
        print(f"    Centroid: phi={a.phi_centroid:.2f}, conflict={a.conflict_centroid:.2f}")
        print(f"    Occurrences: {a.occurrence_count}")
        
    print("\n[Memory Statistics]")
    print("-" * 40)
    
    stats = memory.get_stats()
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Halted actions: {stats['halted_actions']}")
    print(f"  Halt rate: {stats['halt_rate']:.1%}")
    print(f"  Average phi: {stats['average_phi']:.3f}")
    print(f"  Average conflict: {stats['average_conflict']:.3f}")
    
    print("\n" + "=" * 60)
    print("Key features:")
    print("1. Memories persist across restarts")
    print("2. State attractors learned automatically")
    print("3. Similar state retrieval for decision support")
    print("=" * 60)


def demo_mock():
    """Mock demo when database not available"""
    print("\n[Mock Demo - No Database]")
    print("-" * 40)
    print("In production, PersistentMemory:")
    print("  - Stores decisions in PostgreSQL")
    print("  - Learns state attractors (high_confidence, uncertain, etc.)")
    print("  - Finds similar past states for decision support")
    print("  - Survives server restarts")
    print("\nTo run with real database:")
    print("  export DATABASE_URL='postgresql://user:pass@host/db'")
    print("  python persistent_memory.py")


if __name__ == '__main__':
    main()
