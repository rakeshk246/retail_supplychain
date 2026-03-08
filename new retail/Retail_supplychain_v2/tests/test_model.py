# tests/test_model.py
# ==============================================================================
# Integration tests for SupplyChainModel
# Tests: step execution, disruption/recovery, multi-day simulation
# ==============================================================================

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer import DataLayer
from forecasting_module import DemandForecaster
from supply_chain_model import SupplyChainModel


@pytest.fixture
def model():
    """Create a model for integration testing."""
    data_layer = DataLayer()
    data_layer.generate_historical_data(100)
    forecaster = DemandForecaster(method='lstm')
    forecaster.train(data_layer.historical_data, epochs=5)
    return SupplyChainModel(forecaster, data_layer)


class TestSupplyChainModel:
    def test_single_step(self, model):
        """Model should execute a single step without errors."""
        model.step()
        assert model.current_day == 1
        assert model.daily_demand > 0

    def test_multi_step(self, model):
        """Model should run 30 days without errors."""
        for _ in range(30):
            model.step()
        assert model.current_day == 30

    def test_data_collection(self, model):
        """DataCollector should record data each step."""
        for _ in range(10):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        assert len(df) == 10
        assert 'Inventory' in df.columns
        assert 'Demand' in df.columns

    def test_event_logging(self, model):
        """Events should be logged during simulation."""
        model.step()
        assert len(model.event_log) > 0
        assert 'day' in model.event_log[0]
        assert 'agent' in model.event_log[0]
        assert 'message' in model.event_log[0]


class TestDisruptionRecovery:
    """BUG 3 REGRESSION TESTS: Disruption auto-recovery."""

    def test_inject_supplier_disruption(self, model):
        """Supplier should be marked as disrupted."""
        model.step()
        model.inject_disruption('supplier', duration=3)
        assert model.supplier.status == 'disrupted'
        assert 'supplier' in model.disruption_schedule

    def test_supplier_auto_recovery(self, model):
        """Supplier should auto-recover after disruption duration.
        
        This is the core Bug 3 regression test. In the old code,
        disruption never ended — the supplier stayed disrupted forever.
        """
        # Run 1 day first
        model.step()
        assert model.current_day == 1

        # Inject 3-day disruption at day 1
        model.inject_disruption('supplier', duration=3)
        assert model.supplier.status == 'disrupted'

        # Day 2, 3 — still disrupted
        model.step()  # day 2
        assert model.supplier.status == 'disrupted'
        model.step()  # day 3
        assert model.supplier.status == 'disrupted'

        # Day 4 — should recover (end_day = 1 + 3 = 4)
        model.step()  # day 4
        assert model.supplier.status == 'active'
        assert 'supplier' not in model.disruption_schedule

    def test_logistics_auto_recovery(self, model):
        """Logistics should auto-recover after disruption duration."""
        model.step()  # day 1
        model.inject_disruption('logistics', duration=2)
        assert model.logistics.status == 'disrupted'

        model.step()  # day 2 — still disrupted
        assert model.logistics.status == 'disrupted'

        # day 3 — end_day = 1+2 = 3, recovery fires when current_day >= end_day
        model.step()
        assert model.logistics.status == 'active'

    def test_dual_disruption_recovery(self, model):
        """Both supplier and logistics should recover independently."""
        model.step()  # day 1
        model.inject_disruption('supplier', duration=3)
        model.inject_disruption('logistics', duration=5)

        # At day 4, supplier recovers but logistics still disrupted
        for _ in range(3):
            model.step()
        assert model.supplier.status == 'active'
        assert model.logistics.status == 'disrupted'

        # At day 6, logistics recovers
        for _ in range(2):
            model.step()
        assert model.logistics.status == 'active'

    def test_supplier_cannot_fulfill_during_disruption(self, model):
        """Orders during disruption should fail."""
        model.supplier.status = 'disrupted'
        result = model.supplier.process_order(100)
        assert result == 0

    def test_get_status(self, model):
        """get_status() should return complete model state."""
        model.step()
        status = model.get_status()
        assert 'day' in status
        assert 'inventory' in status
        assert 'supplier_status' in status
        assert 'logistics_status' in status
        assert 'pending_shipments' in status
