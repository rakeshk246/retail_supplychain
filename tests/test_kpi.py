# tests/test_kpi.py
# ==============================================================================
# Tests for KPIEvaluator — metrics calculation, recovery time, baseline export
# ==============================================================================

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kpi_evaluator import KPIEvaluator


@pytest.fixture
def kpi():
    return KPIEvaluator()


class TestKPICalculation:
    def test_empty_kpis(self, kpi):
        """KPIs should be empty dict when no data."""
        result = kpi.calculate_kpis()
        assert result == {}

    def test_perfect_fulfillment(self, kpi):
        """100% fill rate when all demand is fulfilled."""
        for i in range(10):
            kpi.update(demand=100, fulfilled=100, inventory=500,
                       stockout=False, day=i + 1)
        result = kpi.calculate_kpis()
        assert result['Fill Rate (%)'] == 100.0
        assert result['Stock-out Rate (%)'] == 0.0

    def test_partial_fulfillment(self, kpi):
        """Fill rate should reflect partial fulfillment."""
        kpi.update(demand=100, fulfilled=80, inventory=0,
                   stockout=True, day=1)
        result = kpi.calculate_kpis()
        assert result['Fill Rate (%)'] == 80.0
        assert result['Stock-out Rate (%)'] == 100.0

    def test_avg_inventory(self, kpi):
        """Average inventory should be correct."""
        kpi.update(demand=50, fulfilled=50, inventory=200,
                   stockout=False, day=1)
        kpi.update(demand=50, fulfilled=50, inventory=400,
                   stockout=False, day=2)
        result = kpi.calculate_kpis()
        assert result['Avg Inventory'] == 300.0

    def test_resilience_index(self, kpi):
        """Resilience index should decrease with more stockouts."""
        for i in range(10):
            kpi.update(demand=100, fulfilled=50, inventory=0,
                       stockout=True, day=i + 1)
        result = kpi.calculate_kpis()
        assert result['Resilience Index'] < 100


class TestRecoveryTime:
    def test_recovery_tracking(self, kpi):
        """Recovery time should be tracked after disruption ends."""
        # Normal days
        for i in range(5):
            kpi.update(demand=100, fulfilled=100, inventory=500,
                       stockout=False, day=i + 1, disruption_active=False)

        # Disruption active (days 6-8)
        for i in range(5, 8):
            kpi.update(demand=100, fulfilled=50, inventory=100,
                       stockout=True, day=i + 1, disruption_active=True)

        # Post-disruption recovery (fill rate < 90%)
        kpi.update(demand=100, fulfilled=85, inventory=200,
                   stockout=False, day=9, disruption_active=False)

        # Full recovery (fill rate >= 90%)
        kpi.update(demand=100, fulfilled=95, inventory=400,
                   stockout=False, day=10, disruption_active=False)

        result = kpi.calculate_kpis()
        assert result['Total Disruptions'] >= 1


class TestDailyTrends:
    def test_daily_snapshots_stored(self, kpi):
        """Daily snapshots should be stored."""
        for i in range(5):
            kpi.update(demand=100, fulfilled=100, inventory=500,
                       stockout=False, day=i + 1)
        assert len(kpi.daily_snapshots) == 5

    def test_get_daily_trend(self, kpi):
        """get_daily_trend should return list of values."""
        for i in range(5):
            kpi.update(demand=100, fulfilled=100, inventory=500,
                       stockout=False, day=i + 1)
        trend = kpi.get_daily_trend('fill_rate')
        assert len(trend) == 5
        assert all(v == 100.0 for v in trend)


class TestBaselineExport:
    def test_export_baseline(self, kpi, tmp_path):
        """export_baseline should create a valid JSON file."""
        for i in range(10):
            kpi.update(demand=100, fulfilled=90, inventory=300,
                       stockout=False, day=i + 1)

        filepath = str(tmp_path / 'test_baseline.json')
        result = kpi.export_baseline(filepath)

        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert 'kpis' in data
        assert 'daily_snapshots' in data
        assert 'timestamp' in data

    def test_reset(self, kpi):
        """reset should clear all metrics."""
        for i in range(5):
            kpi.update(demand=100, fulfilled=100, inventory=500,
                       stockout=False, day=i + 1)
        kpi.reset()
        assert kpi.metrics['total_days'] == 0
        assert len(kpi.daily_snapshots) == 0
