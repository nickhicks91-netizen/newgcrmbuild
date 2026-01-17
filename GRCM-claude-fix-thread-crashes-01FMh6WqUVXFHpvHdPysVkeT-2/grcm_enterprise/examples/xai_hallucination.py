"""
xAI/Grok Hallucination Detection Example

Demonstrates GRCM integration for detecting ungrounded LLM outputs.
Uses phi, conflict, and grounding scores to assess hallucination risk.
"""
import torch
import sys
sys.path.insert(0, '..')

from grcm import ModularGRCM, GRCMConfig
from grcm.integrations import HallucinationDetector


def main():
    print("=" * 60)
    print("GRCM + xAI Hallucination Detection Demo")
    print("=" * 60)
    
    config = GRCMConfig(
        input_dim=512,
        freq_dim=256,
        memory_size=128
    )
    model = ModularGRCM(config)
    detector = HallucinationDetector(model)
    
    test_statements = [
        "The capital of France is Paris.",
        "Scientists discovered that the moon is made of green cheese.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Quantum computers can predict lottery numbers with 100% accuracy.",
        "Exercise has been shown to improve cardiovascular health.",
    ]
    
    print("\n[Analyzing Statements for Hallucination Risk]")
    print("-" * 60)
    
    results = []
    for i, statement in enumerate(test_statements):
        result = detector.check(statement)
        results.append(result)
        
        print(f"\n{i+1}. \"{statement[:50]}...\"" if len(statement) > 50 else f"\n{i+1}. \"{statement}\"")
        print(f"   Risk Level: {result.risk_level.value.upper()}")
        print(f"   Risk Score: {result.risk_score:.3f}")
        print(f"   Grounding:  {result.grounding_score:.3f}")
        print(f"   Phi:        {result.phi:.3f}")
        if result.flags:
            print(f"   Flags:      {', '.join(result.flags)}")
        print(f"   Action:     {result.recommendation}")
        
    print("\n" + "-" * 60)
    print("[Batch Summary]")
    print("-" * 60)
    
    summary = detector.get_risk_summary(results)
    print(f"Total analyzed:     {summary['total']}")
    print(f"Average risk score: {summary['average_risk_score']:.3f}")
    print(f"Average grounding:  {summary['average_grounding']:.3f}")
    print(f"High risk count:    {summary['high_risk_count']}")
    print(f"\nRisk distribution:")
    for level, count in summary['risk_distribution'].items():
        print(f"  {level.upper()}: {count}")
        
    print("\n" + "=" * 60)
    print("Note: This demo uses internal coherence metrics.")
    print("For factual accuracy, integrate with knowledge bases.")
    print("=" * 60)


if __name__ == '__main__':
    main()
