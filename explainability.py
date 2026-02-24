# explainability.py
# ==============================================================================
# Phase 4: Explainable AI (XAI) for Supply Chain Agents
# Every decision records WHY it was made — full reasoning chain
# ==============================================================================

import json
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict


class DecisionRecord:
    """A single explainable decision record.
    
    Captures the full "WHY" behind every agent decision:
    - What was the situation?
    - What did the agent decide?
    - WHY did it make that choice?
    - What factors influenced the decision?
    - What alternatives were considered?
    - How confident was the agent?
    """
    
    def __init__(self, agent: str, decision_type: str, day: int):
        self.agent = agent
        self.decision_type = decision_type
        self.day = day
        self.timestamp = datetime.now().isoformat()
        
        # WHAT happened
        self.action = ""
        self.details = {}
        
        # WHY — the core explainability
        self.reasoning = ""              # LLM's step-by-step reasoning
        self.why_summary = ""            # Human-readable one-line explanation
        self.contributing_factors = []   # What data points drove this decision
        self.alternatives_considered = [] # Other options + why they were rejected
        self.triggered_by = ""           # What triggered this decision
        
        # Context at decision time
        self.context = {}                # Snapshot of state when decision was made
        self.memory_used = []            # Past episodes retrieved from memory
        self.messages_received = []      # Inter-agent messages that influenced decision
        
        # Confidence & workflow
        self.confidence = 0.0
        self.workflow_path = ""          # Which LangGraph path was taken
        self.is_llm_decision = False     # True = LLM decided, False = rule-based fallback
        self.fallback_reason = ""        # If rule-based, why did LLM fail?
    
    def set_why(self, reasoning: str, summary: str,
                factors: List[str] = None, alternatives: List[Dict] = None):
        """Set the WHY explanation for this decision."""
        self.reasoning = reasoning
        self.why_summary = summary
        if factors:
            self.contributing_factors = factors
        if alternatives:
            self.alternatives_considered = alternatives
    
    def to_dict(self) -> dict:
        return {
            'agent': self.agent,
            'decision_type': self.decision_type,
            'day': self.day,
            'timestamp': self.timestamp,
            'action': self.action,
            'details': self.details,
            'why': {
                'reasoning': self.reasoning,
                'summary': self.why_summary,
                'contributing_factors': self.contributing_factors,
                'alternatives_considered': self.alternatives_considered,
                'triggered_by': self.triggered_by
            },
            'context': self.context,
            'confidence': self.confidence,
            'workflow_path': self.workflow_path,
            'is_llm_decision': self.is_llm_decision,
            'fallback_reason': self.fallback_reason
        }
    
    def explain(self) -> str:
        """Generate human-readable explanation of WHY this decision was made."""
        lines = []
        lines.append(f"Day {self.day} | {self.agent} Agent | {self.decision_type}")
        lines.append(f"  Action: {self.action}")
        
        if self.why_summary:
            lines.append(f"  WHY: {self.why_summary}")
        
        if self.triggered_by:
            lines.append(f"  Triggered by: {self.triggered_by}")
        
        if self.contributing_factors:
            lines.append(f"  Key factors:")
            for f in self.contributing_factors:
                lines.append(f"    - {f}")
        
        if self.alternatives_considered:
            lines.append(f"  Alternatives considered:")
            for alt in self.alternatives_considered:
                name = alt.get('option', 'unknown')
                reason = alt.get('rejected_because', '')
                lines.append(f"    - {name}: rejected because {reason}")
        
        if self.reasoning:
            lines.append(f"  Full reasoning: {self.reasoning[:200]}...")
        
        src = "AI (LLM)" if self.is_llm_decision else "Rule-based"
        lines.append(f"  Source: {src} | Confidence: {self.confidence:.0%}")
        
        if self.workflow_path:
            lines.append(f"  Workflow path: {self.workflow_path}")
        
        return "\n".join(lines)


class ExplainabilityEngine:
    """Central engine for tracking and explaining ALL agent decisions.
    
    Provides:
    1. Full decision audit trail
    2. Human-readable explanations of WHY each decision was made
    3. Decision chain visualization (which decisions led to which)
    4. Aggregated stats (confidence, LLM vs rule-based, etc.)
    """
    
    _instance = None
    
    def __init__(self):
        self.decisions = []              # All decision records
        self.decisions_by_day = defaultdict(list)
        self.decisions_by_agent = defaultdict(list)
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        cls._instance = None
    
    def record(self, decision: DecisionRecord):
        """Record a new decision."""
        self.decisions.append(decision)
        self.decisions_by_day[decision.day].append(decision)
        self.decisions_by_agent[decision.agent].append(decision)
    
    def create_record(self, agent: str, decision_type: str, day: int) -> DecisionRecord:
        """Create and register a new decision record."""
        record = DecisionRecord(agent, decision_type, day)
        self.record(record)
        return record
    
    # =========================================================================
    # WHY Explanations
    # =========================================================================
    
    def explain_day(self, day: int) -> str:
        """Get full explanation of all decisions for a given day."""
        records = self.decisions_by_day.get(day, [])
        if not records:
            return f"No decisions recorded for day {day}."
        
        lines = [f"=== Day {day} Decision Chain ===\n"]
        for i, rec in enumerate(records, 1):
            lines.append(f"--- Decision {i} ---")
            lines.append(rec.explain())
            lines.append("")
        
        return "\n".join(lines)
    
    def explain_agent(self, agent: str, last_n: int = 5) -> str:
        """Get recent decisions by a specific agent with WHY explanations."""
        records = self.decisions_by_agent.get(agent, [])
        if not records:
            return f"No decisions recorded for {agent}."
        
        recent = records[-last_n:]
        lines = [f"=== {agent} Agent — Last {len(recent)} Decisions ===\n"]
        for rec in recent:
            lines.append(rec.explain())
            lines.append("")
        
        return "\n".join(lines)
    
    def get_decision_chain(self, day: int) -> List[Dict]:
        """Get the decision chain for a day as structured data."""
        records = self.decisions_by_day.get(day, [])
        return [r.to_dict() for r in records]
    
    def get_latest_decisions(self, n: int = 10) -> List[Dict]:
        """Get the most recent decisions."""
        return [r.to_dict() for r in self.decisions[-n:]]
    
    # =========================================================================
    # Analytics
    # =========================================================================
    
    def get_summary(self) -> dict:
        """Get aggregated decision statistics."""
        if not self.decisions:
            return {'total_decisions': 0}
        
        llm_count = sum(1 for d in self.decisions if d.is_llm_decision)
        avg_confidence = sum(d.confidence for d in self.decisions) / len(self.decisions)
        
        by_type = defaultdict(int)
        by_agent = defaultdict(int)
        for d in self.decisions:
            by_type[d.decision_type] += 1
            by_agent[d.agent] += 1
        
        return {
            'total_decisions': len(self.decisions),
            'llm_decisions': llm_count,
            'rule_based_decisions': len(self.decisions) - llm_count,
            'llm_ratio': round(llm_count / len(self.decisions), 2),
            'avg_confidence': round(avg_confidence, 2),
            'by_type': dict(by_type),
            'by_agent': dict(by_agent),
            'days_covered': len(self.decisions_by_day)
        }
    
    def get_confidence_trend(self) -> List[float]:
        """Get confidence values over time."""
        return [d.confidence for d in self.decisions]
    
    def get_why_summaries(self, last_n: int = 20) -> List[Dict]:
        """Get recent WHY summaries — quick overview of decision reasoning."""
        results = []
        for d in self.decisions[-last_n:]:
            results.append({
                'day': d.day,
                'agent': d.agent,
                'action': d.action,
                'why': d.why_summary,
                'confidence': d.confidence,
                'is_llm': d.is_llm_decision,
                'path': d.workflow_path
            })
        return results
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_audit_trail(self, filepath: str = 'xai_audit_trail.json'):
        """Export full audit trail to JSON."""
        data = {
            'summary': self.get_summary(),
            'decisions': [d.to_dict() for d in self.decisions]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return filepath
    
    def format_for_dashboard(self, day: int = None) -> dict:
        """Format data for Streamlit dashboard."""
        if day:
            records = self.decisions_by_day.get(day, [])
        else:
            records = self.decisions[-20:]
        
        return {
            'decisions': [r.to_dict() for r in records],
            'summary': self.get_summary(),
            'confidence_trend': self.get_confidence_trend()
        }
