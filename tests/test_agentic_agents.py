# tests/test_agentic_agents.py
# ==============================================================================
# Tests for Phase 2: Agentic Agents
# Tests LLM fallback behavior, agent decisions, and model integration
# Updated: Tests disable API key to avoid GroqRateLimitError
# ==============================================================================

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer import DataLayer
from forecasting_module import DemandForecaster
from agentic_agents import (
    AgenticSupplierAgent, AgenticWarehouseAgent,
    AgenticLogisticsAgent, AgenticDemandAgent,
    LLMEngine, GroqRateLimitError
)
from agentic_model import AgenticSupplyChainModel


@pytest.fixture(autouse=True)
def reset_llm_and_disable_api():
    """Reset LLM singleton and disable API key for testing.
    
    This ensures tests run without hitting the Groq API,
    testing the pure fallback behavior.
    """
    LLMEngine.reset()
    # Save and remove API key
    saved_key = os.environ.pop('GROQ_API_KEY', None)
    yield
    LLMEngine.reset()
    # Restore API key
    if saved_key:
        os.environ['GROQ_API_KEY'] = saved_key


@pytest.fixture
def model():
    """Create agentic model for testing (rule_based mode = no API calls)."""
    data_layer = DataLayer()
    data_layer.generate_historical_data(60)
    forecaster = DemandForecaster(method='lstm')
    forecaster.train(data_layer.historical_data, epochs=3)
    return AgenticSupplyChainModel(forecaster, data_layer, agent_mode='rule_based')


@pytest.fixture
def agentic_model():
    """Create model in agentic mode.
    
    API key is removed by autouse fixture, so LLM won't be available.
    Agents should use smart fallback logic.
    """
    data_layer = DataLayer()
    data_layer.generate_historical_data(60)
    forecaster = DemandForecaster(method='lstm')
    forecaster.train(data_layer.historical_data, epochs=3)
    return AgenticSupplyChainModel(forecaster, data_layer, agent_mode='agentic')


# =============================================================================
# LLM Engine Tests
# =============================================================================

class TestLLMEngine:
    def test_singleton(self):
        """LLM engine should be a singleton."""
        engine1 = LLMEngine.get_instance()
        engine2 = LLMEngine.get_instance()
        assert engine1 is engine2

    def test_reset_creates_new_instance(self):
        """Reset should create a new instance."""
        engine1 = LLMEngine.get_instance()
        LLMEngine.reset()
        engine2 = LLMEngine.get_instance()
        assert engine1 is not engine2

    def test_no_api_key_fallback(self):
        """Without API key, LLM should be unavailable."""
        LLMEngine.reset()
        engine = LLMEngine.get_instance()
        assert not engine.is_available

    def test_reason_returns_none_without_llm(self):
        """reason() should return None when LLM is unavailable."""
        engine = LLMEngine.get_instance()
        assert not engine.is_available
        result = engine.reason("system", "question")
        assert result is None

    def test_stats(self):
        """get_stats should return valid structure."""
        engine = LLMEngine.get_instance()
        stats = engine.get_stats()
        assert 'model' in stats
        assert 'available' in stats
        assert 'total_calls' in stats
        assert 'rate_limited' in stats

    def test_rate_limit_flag(self):
        """Rate limit flag should work correctly."""
        engine = LLMEngine.get_instance()
        assert not engine.rate_limited
        # Simulate rate limit
        engine.rate_limited = True
        engine.rate_limit_message = "Test rate limit"
        assert not engine.is_available
        with pytest.raises(GroqRateLimitError):
            engine.reason("system", "question")


# =============================================================================
# Agentic Agent Fallback Tests
# =============================================================================

class TestAgenticSupplierFallback:
    """Test that agentic supplier uses smart fallback when no LLM."""

    def test_process_order_fallback(self, agentic_model):
        """Without LLM, supplier should use rule-based logic."""
        result = agentic_model.supplier.process_order(100)
        assert isinstance(result, dict)
        assert result['quantity'] > 0

    def test_process_order_disrupted(self, agentic_model):
        """Disrupted supplier should return 0 regardless of mode."""
        agentic_model.supplier.status = 'disrupted'
        result = agentic_model.supplier.process_order(100)
        assert result == 0


class TestAgenticWarehouseFallback:
    def test_fulfill_demand(self, agentic_model):
        """Warehouse should still fulfill demand without LLM."""
        warehouse = agentic_model.warehouse
        warehouse.inventory = 500
        fulfilled = warehouse.fulfill_demand(100)
        assert fulfilled == 100
        assert warehouse.inventory == 400

    def test_check_reorder_fallback(self, agentic_model):
        """Without LLM, reorder should use smart fallback logic."""
        warehouse = agentic_model.warehouse
        warehouse.inventory = 100
        order_qty = warehouse.check_reorder(100)
        assert order_qty > 0

    def test_demand_tracking(self, agentic_model):
        """Warehouse should track demand history."""
        warehouse = agentic_model.warehouse
        warehouse.fulfill_demand(100)
        warehouse.fulfill_demand(150)
        assert len(warehouse.demand_history) == 2


class TestAgenticLogisticsFallback:
    def test_schedule_shipment(self, agentic_model):
        """Logistics should schedule shipments without LLM."""
        logistics = agentic_model.logistics
        agentic_model.current_day = 5
        logistics.schedule_shipment(quantity=100, lead_time=3)
        assert len(logistics.shipments) == 1
        assert logistics.shipments[0]['arrival_day'] == 8

    def test_delivery(self, agentic_model):
        """Shipments should deliver on arrival."""
        model = agentic_model
        model.current_day = 1
        model.logistics.schedule_shipment(100, 2)
        model.current_day = 3
        model.logistics.step()
        assert len(model.logistics.shipments) == 0


class TestAgenticDemandFallback:
    def test_demand_generation(self, agentic_model):
        """Demand agent should generate demand without LLM."""
        agentic_model.demand_agent.step()
        assert agentic_model.daily_demand > 0


# =============================================================================
# Model Mode Tests
# =============================================================================

class TestAgenticModel:
    def test_rule_based_mode(self, model):
        """Rule-based mode should work without LLM."""
        model.step()
        assert model.current_day == 1
        assert model.daily_demand > 0

    def test_agentic_mode_fallback(self, agentic_model):
        """Agentic mode should work with smart fallback when no API key."""
        agentic_model.step()
        assert agentic_model.current_day == 1

    def test_multi_step(self, agentic_model):
        """Agentic model should run 15 days without errors."""
        for _ in range(15):
            agentic_model.step()
        assert agentic_model.current_day == 15

    def test_disruption_recovery(self, agentic_model):
        """Disruption recovery should work in agentic mode."""
        agentic_model.step()
        agentic_model.inject_disruption('supplier', 2)
        assert agentic_model.supplier.status == 'disrupted'
        agentic_model.step()
        agentic_model.step()
        assert agentic_model.supplier.status == 'active'

    def test_get_status(self, agentic_model):
        """get_status should return mode info."""
        agentic_model.step()
        status = agentic_model.get_status()
        assert status['mode'] == 'agentic'
        assert 'llm_stats' in status

    def test_get_agent_reasoning(self, agentic_model):
        """get_agent_reasoning should return dict."""
        agentic_model.step()
        reasoning = agentic_model.get_agent_reasoning()
        assert isinstance(reasoning, dict)

    def test_data_collection(self, agentic_model):
        """DataCollector should include Agent_Mode."""
        for _ in range(5):
            agentic_model.step()
        df = agentic_model.datacollector.get_model_vars_dataframe()
        assert len(df) == 5
        assert 'Agent_Mode' in df.columns
