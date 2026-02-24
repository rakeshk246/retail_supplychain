# tests/test_orchestrator.py
# ==============================================================================
# Tests for Phase 4: LangGraph Orchestration + Explainable AI
# ==============================================================================

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer import DataLayer
from forecasting_module import DemandForecaster
from orchestrator import OrchestratedSupplyChainModel
from explainability import ExplainabilityEngine, DecisionRecord
from agentic_agents import LLMEngine
from message_bus import MessageBus


@pytest.fixture(autouse=True)
def reset_singletons():
    LLMEngine.reset()
    MessageBus.reset()
    ExplainabilityEngine.reset()
    # Remove API key to prevent rate limit errors during tests
    saved_key = os.environ.pop('GROQ_API_KEY', None)
    yield
    LLMEngine.reset()
    MessageBus.reset()
    ExplainabilityEngine.reset()
    if saved_key:
        os.environ['GROQ_API_KEY'] = saved_key


@pytest.fixture
def model():
    data_layer = DataLayer()
    data_layer.generate_historical_data(60)
    forecaster = DemandForecaster(method='lstm')
    forecaster.train(data_layer.historical_data, epochs=3)
    return OrchestratedSupplyChainModel(
        forecaster, data_layer, agent_mode='rule_based'
    )


@pytest.fixture
def agentic_model():
    data_layer = DataLayer()
    data_layer.generate_historical_data(60)
    forecaster = DemandForecaster(method='lstm')
    forecaster.train(data_layer.historical_data, epochs=3)
    return OrchestratedSupplyChainModel(
        forecaster, data_layer, agent_mode='agentic'
    )


# =============================================================================
# Explainability Engine Tests
# =============================================================================

class TestExplainability:
    def test_decision_record(self):
        rec = DecisionRecord("Supplier", "order", day=5)
        rec.action = "Ship 300 units"
        rec.set_why(
            reasoning="Order within capacity and reliability is high",
            summary="Fully fulfilled because capacity allows it",
            factors=["Capacity=500", "Order=300"],
            alternatives=[
                {'option': 'Partial fill', 'rejected_because': 'no capacity issue'}
            ]
        )
        rec.confidence = 0.9
        assert rec.why_summary == "Fully fulfilled because capacity allows it"
        assert len(rec.contributing_factors) == 2
        assert len(rec.alternatives_considered) == 1

    def test_explain_output(self):
        rec = DecisionRecord("Warehouse", "reorder", day=3)
        rec.action = "Emergency reorder 500"
        rec.set_why(
            reasoning="Inventory below critical threshold",
            summary="Emergency reorder because only 2 days of stock left",
            factors=["Inventory=100", "Demand=50/day"]
        )
        rec.confidence = 0.95
        text = rec.explain()
        assert "WHY" in text
        assert "Emergency" in text
        assert "Inventory=100" in text

    def test_engine_record_and_retrieve(self):
        engine = ExplainabilityEngine()
        rec = engine.create_record("Demand", "forecast", day=1)
        rec.action = "Forecast 120 units"
        rec.set_why(reasoning="trend", summary="Increasing trend detected")
        assert engine.get_summary()['total_decisions'] == 1

    def test_explain_day(self):
        engine = ExplainabilityEngine()
        r1 = engine.create_record("Demand", "forecast", day=5)
        r1.action = "Forecast 100"
        r1.set_why(reasoning="stable", summary="Stable demand")
        r2 = engine.create_record("Warehouse", "reorder", day=5)
        r2.action = "Reorder 300"
        r2.set_why(reasoning="low stock", summary="Low inventory")
        text = engine.explain_day(5)
        assert "Demand" in text
        assert "Warehouse" in text

    def test_why_summaries(self):
        engine = ExplainabilityEngine()
        r = engine.create_record("Supplier", "order", day=1)
        r.action = "Ship 200"
        r.set_why(reasoning="ok", summary="Within capacity")
        summaries = engine.get_why_summaries()
        assert len(summaries) == 1
        assert summaries[0]['why'] == "Within capacity"

    def test_confidence_trend(self):
        engine = ExplainabilityEngine()
        for i in range(5):
            r = engine.create_record("Test", "action", day=i)
            r.confidence = 0.5 + i * 0.1
        trend = engine.get_confidence_trend()
        assert len(trend) == 5
        assert trend[-1] == pytest.approx(0.9)

    def test_to_dict(self):
        rec = DecisionRecord("Agent", "type", day=1)
        rec.set_why(reasoning="because", summary="summary", factors=["f1"])
        d = rec.to_dict()
        assert d['why']['summary'] == "summary"
        assert d['why']['contributing_factors'] == ["f1"]


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestOrchestrator:
    def test_single_step(self, model):
        model.step()
        assert model.current_day == 1

    def test_multi_step(self, model):
        for _ in range(10):
            model.step()
        assert model.current_day == 10

    def test_workflow_path_tracked(self, model):
        model.step()
        status = model.get_status()
        assert 'workflow_path' in status
        assert status['workflow_path'] in ('NORMAL', 'EMERGENCY', 'CRISIS', 'none')

    def test_xai_decisions_recorded(self, model):
        for _ in range(5):
            model.step()
        decisions = model.get_xai_decisions(last_n=50)
        assert len(decisions) > 0

    def test_xai_has_why(self, model):
        model.step()
        decisions = model.get_xai_decisions(last_n=10)
        for d in decisions:
            assert 'why' in d
            assert 'summary' in d['why']

    def test_explain_day(self, model):
        model.step()
        explanation = model.explain_day(1)
        assert "Day 1" in explanation
        assert "WHY" in explanation

    def test_disruption_changes_path(self, model):
        """Disruption should trigger emergency/crisis path."""
        for _ in range(3):
            model.step()
        model.inject_disruption('supplier', 3)
        # Drain inventory to trigger emergency
        model.warehouse.inventory = 50
        model.step()
        # Should have taken emergency or crisis path
        assert model._last_workflow_path in ('EMERGENCY', 'CRISIS')

    def test_disruption_recovery(self, model):
        model.step()
        model.inject_disruption('supplier', 2)
        assert model.supplier.status == 'disrupted'
        model.step()
        model.step()
        assert model.supplier.status == 'active'

    def test_agentic_mode(self, agentic_model):
        agentic_model.step()
        assert agentic_model.current_day == 1

    def test_get_status_includes_xai(self, model):
        model.step()
        status = model.get_status()
        assert 'xai_summary' in status

    def test_data_collection(self, model):
        for _ in range(5):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        assert len(df) == 5
        assert 'Workflow_Path' in df.columns

    def test_crisis_path_on_stockout(self, model):
        """Stockout should trigger crisis path."""
        model.warehouse.inventory = 0
        model.inject_disruption('supplier', 3)
        model.step()
        assert model._last_workflow_path == 'CRISIS'

    def test_normal_path_high_inventory(self, model):
        """High inventory should take normal path."""
        model.warehouse.inventory = 800
        model.step()
        assert model._last_workflow_path == 'NORMAL'
