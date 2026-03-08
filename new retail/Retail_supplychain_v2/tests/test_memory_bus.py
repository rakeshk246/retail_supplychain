# tests/test_memory_bus.py
# ==============================================================================
# Tests for Phase 3: Memory + Message Bus
# ==============================================================================

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import AgentMemory
from message_bus import MessageBus, Message
from agentic_agents import LLMEngine
from agentic_model import AgenticSupplyChainModel
from data_layer import DataLayer
from forecasting_module import DemandForecaster


@pytest.fixture(autouse=True)
def reset_singletons():
    LLMEngine.reset()
    MessageBus.reset()
    # Remove API key to prevent rate limit errors during tests
    saved_key = os.environ.pop('GROQ_API_KEY', None)
    yield
    LLMEngine.reset()
    MessageBus.reset()
    if saved_key:
        os.environ['GROQ_API_KEY'] = saved_key


# =============================================================================
# Memory Tests
# =============================================================================

class TestAgentMemory:
    def test_store_and_count(self):
        mem = AgentMemory("TestAgent")
        mem.store_episode("low stock", "reorder 500", {'success': True}, day=1)
        mem.store_episode("high stock", "hold", {'success': True}, day=2)
        assert mem.get_episode_count() >= 2

    def test_recall_similar(self):
        mem = AgentMemory("TestAgent")
        mem.store_episode("inventory at 50, demand high",
                          "emergency reorder", {'success': True}, day=1)
        mem.store_episode("inventory at 800, demand low",
                          "hold stock", {'success': True}, day=2)
        results = mem.recall_similar("inventory very low, high demand", n_results=1)
        assert len(results) >= 1

    def test_recall_empty(self):
        mem = AgentMemory("EmptyAgent")
        results = mem.recall_similar("anything", n_results=3)
        assert len(results) == 0

    def test_format_for_prompt(self):
        mem = AgentMemory("TestAgent")
        mem.store_episode("situation A", "decision A", {'success': True}, day=1)
        episodes = mem.recall_similar("situation A", n_results=1)
        text = mem.format_for_prompt(episodes)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_success_rate(self):
        mem = AgentMemory("TestAgent")
        mem.store_episode("s1", "d1", {'success': True}, day=1)
        mem.store_episode("s2", "d2", {'success': False}, day=2)
        mem.store_episode("s3", "d3", {'success': True}, day=3)
        rate = mem.get_success_rate()
        assert abs(rate - 2/3) < 0.01

    def test_clear(self):
        mem = AgentMemory("TestAgent")
        mem.store_episode("s1", "d1", {'success': True}, day=1)
        mem.clear()
        assert mem.get_episode_count() == 0 or len(mem.episodes) == 0

    def test_get_stats(self):
        mem = AgentMemory("TestAgent")
        stats = mem.get_stats()
        assert stats['agent'] == 'TestAgent'
        assert 'total_episodes' in stats
        assert 'success_rate' in stats


# =============================================================================
# Message Bus Tests
# =============================================================================

class TestMessageBus:
    def test_publish_and_get(self):
        bus = MessageBus()
        msg = Message("Supplier", "Warehouse", "order_confirmation",
                      {'quantity': 100})
        bus.publish(msg)
        messages = bus.get_messages("Warehouse")
        assert len(messages) == 1
        assert messages[0].content['quantity'] == 100

    def test_broadcast(self):
        bus = MessageBus()
        # Initialize inboxes
        bus.inbox['Warehouse'] = []
        bus.inbox['Logistics'] = []
        bus.broadcast_alert("Demand", "demand_forecast",
                            {'predicted': 200, 'trend': 'increasing'})
        # Both agents should receive the broadcast
        assert len(bus.get_messages("Warehouse")) == 1
        assert len(bus.get_messages("Logistics")) == 1

    def test_direct_message(self):
        bus = MessageBus()
        bus.send_direct("Warehouse", "Supplier", "order_request",
                        {'quantity': 300, 'urgency': 'high'})
        msgs = bus.get_messages("Supplier")
        assert len(msgs) == 1
        assert msgs[0].content['urgency'] == 'high'

    def test_message_type_filter(self):
        bus = MessageBus()
        bus.send_direct("A", "B", "type1", {'data': 1})
        bus.send_direct("A", "B", "type2", {'data': 2})
        bus.send_direct("A", "B", "type1", {'data': 3})
        msgs = bus.get_messages("B", msg_type="type1")
        assert len(msgs) == 2

    def test_get_latest(self):
        bus = MessageBus()
        bus.send_direct("A", "B", "alert", {'value': 1})
        bus.send_direct("A", "B", "alert", {'value': 2})
        latest = bus.get_latest("B", "alert")
        assert latest is not None
        assert latest.content['value'] == 2

    def test_format_for_prompt(self):
        bus = MessageBus()
        bus.send_direct("Supplier", "Warehouse", "order_confirmation",
                        {'quantity': 100})
        text = bus.format_for_prompt("Warehouse")
        assert "Supplier" in text
        assert "100" in text

    def test_get_stats(self):
        bus = MessageBus()
        bus.send_direct("A", "B", "test", {'data': 1})
        bus.broadcast_alert("C", "alert", {'data': 2})
        stats = bus.get_stats()
        assert stats['total_messages'] == 2
        assert 'test' in stats['message_types']

    def test_clear_inbox(self):
        bus = MessageBus()
        bus.send_direct("A", "B", "test", {'data': 1})
        bus.clear_inbox("B")
        assert len(bus.get_messages("B")) == 0

    def test_message_serialization(self):
        msg = Message("Supplier", "Warehouse", "order", {'qty': 50})
        d = msg.to_dict()
        msg2 = Message.from_dict(d)
        assert msg2.sender == "Supplier"
        assert msg2.content['qty'] == 50


# =============================================================================
# Integration: Memory + Bus with Model
# =============================================================================

class TestPhase3Integration:
    @pytest.fixture
    def model(self):
        data_layer = DataLayer()
        data_layer.generate_historical_data(60)
        forecaster = DemandForecaster(method='lstm')
        forecaster.train(data_layer.historical_data, epochs=3)
        return AgenticSupplyChainModel(forecaster, data_layer, agent_mode='agentic')

    def test_agents_have_memory(self, model):
        assert hasattr(model.supplier, 'memory')
        assert hasattr(model.warehouse, 'memory')
        assert hasattr(model.logistics, 'memory')
        assert hasattr(model.demand_agent, 'memory')

    def test_agents_have_bus(self, model):
        assert hasattr(model.supplier, 'bus')
        assert hasattr(model.warehouse, 'bus')

    def test_memory_fills_during_simulation(self, model):
        for _ in range(10):
            model.step()
        # After 10 steps, agents should have stored episodes
        stats = model.get_memory_stats()
        # At least demand agent should have episodes (it runs every step)
        assert stats['Demand']['total_episodes'] > 0

    def test_messages_sent_during_simulation(self, model):
        for _ in range(10):
            model.step()
        log = model.get_message_log(50)
        assert len(log) > 0

    def test_get_memory_stats(self, model):
        model.step()
        stats = model.get_memory_stats()
        assert 'Demand' in stats
        assert 'total_episodes' in stats['Demand']

    def test_bus_stats_in_status(self, model):
        model.step()
        status = model.get_status()
        assert 'bus_stats' in status
        assert 'total_messages' in status['bus_stats']
