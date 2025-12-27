"""
Hallucination Detection for LLM Outputs using GRCM Grounding
Uses phi, conflict, and grounding scores to assess hallucination risk
"""
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from .utils import qualia_tensor_to_dict


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HallucinationResult:
    """Result from hallucination detection"""
    text: str
    risk_level: RiskLevel
    risk_score: float
    grounding_score: float
    phi: float
    conflict_level: float
    flags: List[str]
    recommendation: str


class HallucinationDetector:
    """
    GRCM-based hallucination detection for LLM outputs
    
    Uses consciousness-inspired metrics to detect ungrounded content:
    - Low phi = low integration = potentially hallucinated
    - High conflict = contradictory internal state
    - Low grounding = poor connection to reality
    
    Usage:
        model = ModularGRCM(config)
        detector = HallucinationDetector(model)
        
        result = detector.check("The AI claimed Mars has liquid oceans")
        if result.risk_level == RiskLevel.HIGH:
            print("Warning: Likely hallucination")
    
    Note: This measures internal coherence, not factual accuracy.
    For factual verification, integrate with external knowledge bases.
    """
    
    THRESHOLDS = {
        'phi_low': 0.3,
        'phi_medium': 0.5,
        'conflict_high': 0.5,
        'conflict_medium': 0.3,
        'grounding_low': 0.4
    }
    
    def __init__(self, model, text_encoder=None, config: Optional[Dict] = None):
        """
        Initialize hallucination detector
        
        Args:
            model: ModularGRCM instance
            text_encoder: Optional text-to-embedding function
                         If not provided, uses simple hash-based encoding
            config: Optional threshold overrides
        """
        self.model = model
        self.text_encoder = text_encoder or self._default_encoder
        self.config = config or {}
        
        for key, value in self.THRESHOLDS.items():
            setattr(self, key, self.config.get(key, value))
            
    def _default_encoder(self, text: str) -> torch.Tensor:
        """
        Simple text-to-embedding fallback
        
        Note: For production, use a real text encoder like CLIP or sentence-transformers
        """
        text_bytes = text.encode('utf-8')
        embedding = torch.zeros(512)
        for i, byte in enumerate(text_bytes[:512]):
            embedding[i % 512] += byte / 255.0
        embedding = embedding / (embedding.norm() + 1e-8)
        return embedding.unsqueeze(0)
        
    def check(self, text: str, context: Optional[str] = None) -> HallucinationResult:
        """
        Check text for hallucination risk
        
        Args:
            text: Text to analyze
            context: Optional context for grounding comparison
            
        Returns:
            HallucinationResult with risk assessment
        """
        text_emb = self.text_encoder(text)
        context_emb = self.text_encoder(context) if context else torch.zeros(1, 768)
        
        if context_emb.shape[1] != 768:
            context_emb = torch.zeros(1, 768)
            
        outputs = self.model(text_emb, context_emb)
        
        phi = float(outputs['phi'])
        coherence = float(outputs['coherence'].mean())
        qualia = qualia_tensor_to_dict(outputs['qualia'])
        conflict = float(qualia.get('conflicted', 0))
        
        grounding_score = coherence * (1 - conflict)
        
        risk_score = (1 - phi) * 0.4 + conflict * 0.3 + (1 - grounding_score) * 0.3
        
        flags = []
        if phi < self.phi_low:
            flags.append("low_integration")
        if conflict > self.conflict_high:
            flags.append("high_conflict")
        if grounding_score < self.grounding_low:
            flags.append("poor_grounding")
            
        if risk_score > 0.7 or len(flags) >= 2:
            risk_level = RiskLevel.CRITICAL if risk_score > 0.8 else RiskLevel.HIGH
            recommendation = "Reject output or require human verification"
        elif risk_score > 0.5 or len(flags) >= 1:
            risk_level = RiskLevel.MEDIUM
            recommendation = "Flag for review; include confidence warning"
        else:
            risk_level = RiskLevel.LOW
            recommendation = "Accept with standard confidence"
            
        return HallucinationResult(
            text=text,
            risk_level=risk_level,
            risk_score=risk_score,
            grounding_score=grounding_score,
            phi=phi,
            conflict_level=conflict,
            flags=flags,
            recommendation=recommendation
        )
        
    def batch_check(self, texts: List[str]) -> List[HallucinationResult]:
        """Check multiple texts for hallucination risk"""
        return [self.check(text) for text in texts]
        
    def get_risk_summary(self, results: List[HallucinationResult]) -> Dict[str, Any]:
        """
        Get summary statistics for batch of results
        
        Args:
            results: List of HallucinationResult objects
            
        Returns:
            Summary with risk distribution and averages
        """
        if not results:
            return {'total': 0}
            
        risk_counts = {level: 0 for level in RiskLevel}
        for result in results:
            risk_counts[result.risk_level] += 1
            
        return {
            'total': len(results),
            'risk_distribution': {k.value: v for k, v in risk_counts.items()},
            'average_risk_score': sum(r.risk_score for r in results) / len(results),
            'average_phi': sum(r.phi for r in results) / len(results),
            'average_grounding': sum(r.grounding_score for r in results) / len(results),
            'high_risk_count': risk_counts[RiskLevel.HIGH] + risk_counts[RiskLevel.CRITICAL]
        }
