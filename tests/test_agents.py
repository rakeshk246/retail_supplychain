# tests/test_agents.py
# ==============================================================================
# Unit tests for all supply chain agents
# Tests: SupplierAgent, WarehouseAgent, LogisticsAgent, DemandAgent
# ==============================================================================

import sys
import os
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesa import Model
from mesa.time import RandomActivation
from data_layer import DataLayer
from forecasting_module import DemandForecaster
from supply_chain_model import SupplyChainModel
from agents import SupplierAgent, WarehouseAgent, LogisticsAgent, DemandAgent


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_model():
    """Create a basic model for testing agents."""
    data_layer = DataLayer()
    data_layer.generate_historical_data(60)
    forecaster = DemandForecaster(method='lstm')
    forecaster.train(data_layer.historical_data, epochs=5)
    model = SupplyChainModel(forecaster, data_layer)
    return model


# =============================================================================
# Supplier Agent Tests
# =============================================================================

class TestSupplierAgent:
    def test_process_order_active(self, basic_model):
        """Supplier should process orders when active."""
        supplier = basic_model.supplier
        assert supplier.status == "active"
        result = supplier.process_order(100)
        assert isinstance(result, dict)
        assert result['quantity'] > 0
        assert 'lead_time' in result

    def test_process_order_disrupted(self, basic_model):
        """Supplier should reject orders when disrupted."""
        supplier = basic_model.supplier
        supplier.status = "disrupted"
        result = supplier.process_order(100)
        assert result == 0

    def test_capacity_limit(self, basic_model):
        """Supplier should cap orders at capacity."""
        supplier = basic_model.supplier
        supplier.capacity = 200
        result = supplier.process_order(500)
        assert result['quantity'] <= 200


# =============================================================================
# Warehouse Agent Tests
# =============================================================================

class TestWarehouseAgent:
    def test_fulfill_demand_sufficient(self, basic_model):
        """Warehouse should fulfill full demand when stock is sufficient."""
        warehouse = basic_model.warehouse
        warehouse.inventory = 500
        fulfilled = warehouse.fulfill_demand(100)
        assert fulfilled == 100
        assert warehouse.inventory == 400

    def test_fulfill_demand_insufficient(self, basic_model):
        """Warehouse should partially fulfill when stock is insufficient."""
        warehouse = basic_model.warehouse
        warehouse.inventory = 50
        fulfilled = warehouse.fulfill_demand(100)
        assert fulfilled == 50
        assert warehouse.inventory == 0
        assert basic_model.stockouts >= 1

    def test_check_reorder_triggered(self, basic_model):
        """Reorder should trigger when inventory < reorder point."""
        warehouse = basic_model.warehouse
        warehouse.inventory = 100  # Below default reorder_point of 200
        order_qty = warehouse.check_reorder(predicted_demand=100)
        assert order_qty > 0

    def test_check_reorder_not_triggered(self, basic_model):
        """Reorder should NOT trigger when inventory >= reorder point."""
        warehouse = basic_model.warehouse
        warehouse.inventory = 500  # Above default reorder_point of 200
        order_qty = warehouse.check_reorder(predicted_demand=100)
        assert order_qty == 0

    def test_receive_shipment(self, basic_model):
        """Warehouse should receive shipments up to capacity."""
        warehouse = basic_model.warehouse
        warehouse.inventory = 800
        warehouse.max_capacity = 1000
        warehouse.receive_shipment(300)
        assert warehouse.inventory == 1000  # Capped at max_capacity


# =============================================================================
# Logistics Agent Tests — BUG 4 Regression Tests
# =============================================================================

class TestLogisticsAgent:
    def test_schedule_shipment_with_lead_time(self, basic_model):
        """Shipment should be scheduled with correct arrival day."""
        logistics = basic_model.logistics
        basic_model.current_day = 5
        logistics.schedule_shipment(quantity=100, lead_time=3)
        assert len(logistics.shipments) == 1
        assert logistics.shipments[0]['arrival_day'] == 8
        assert logistics.shipments[0]['quantity'] == 100

    def test_schedule_shipment_dict_input(self, basic_model):
        """Shipment should accept dict input (from supplier)."""
        logistics = basic_model.logistics
        basic_model.current_day = 10
        order = {'quantity': 200, 'lead_time': 4}
        logistics.schedule_shipment(order)
        assert len(logistics.shipments) == 1
        assert logistics.shipments[0]['arrival_day'] == 14
        assert logistics.shipments[0]['quantity'] == 200

    def test_delivery_on_arrival_day(self, basic_model):
        """Shipments should be delivered when arrival_day is reached."""
        logistics = basic_model.logistics
        warehouse = basic_model.warehouse
        initial_inventory = warehouse.inventory

        basic_model.current_day = 1
        logistics.schedule_shipment(quantity=100, lead_time=3)

        # Day 2, 3 — not arrived yet
        basic_model.current_day = 3
        logistics.step()
        assert len(logistics.shipments) == 1  # Still pending

        # Day 4 — arrival!
        basic_model.current_day = 4
        logistics.step()
        assert len(logistics.shipments) == 0  # Delivered
        assert warehouse.inventory == initial_inventory + 100

    def test_disrupted_logistics_adds_delay(self, basic_model):
        """Disrupted logistics should add 2 days to lead time."""
        logistics = basic_model.logistics
        logistics.status = "disrupted"
        basic_model.current_day = 5
        logistics.schedule_shipment(quantity=100, lead_time=3)
        # 3 base + 2 disruption delay = 5 total lead time
        assert logistics.shipments[0]['arrival_day'] == 10

    def test_multiple_shipments_partial_delivery(self, basic_model):
        """Multiple shipments should be delivered independently."""
        logistics = basic_model.logistics
        warehouse = basic_model.warehouse
        initial_inventory = warehouse.inventory

        basic_model.current_day = 1
        logistics.schedule_shipment(quantity=100, lead_time=2)  # Arrives day 3
        logistics.schedule_shipment(quantity=200, lead_time=5)  # Arrives day 6

        # Day 3 — first arrives
        basic_model.current_day = 3
        logistics.step()
        assert len(logistics.shipments) == 1  # One pending
        assert warehouse.inventory == initial_inventory + 100

        # Day 6 — second arrives
        basic_model.current_day = 6
        logistics.step()
        assert len(logistics.shipments) == 0
        assert warehouse.inventory == initial_inventory + 300


# =============================================================================
# Demand Agent Tests
# =============================================================================

class TestDemandAgent:
    def test_demand_generation(self, basic_model):
        """Demand agent should set daily_demand on the model."""
        basic_model.demand_agent.step()
        assert basic_model.daily_demand > 0

    def test_demand_is_positive(self, basic_model):
        """Demand should always be positive."""
        for _ in range(10):
            basic_model.demand_agent.step()
            assert basic_model.daily_demand >= 1
