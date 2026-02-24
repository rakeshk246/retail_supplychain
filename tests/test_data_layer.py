# tests/test_data_layer.py
# ==============================================================================
# Tests for DataLayer — synthetic data generation and M5 loader
# ==============================================================================

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer import DataLayer


class TestSyntheticData:
    def test_generate_returns_dataframe(self):
        """generate_historical_data should return a DataFrame."""
        dl = DataLayer()
        data = dl.generate_historical_data(100)
        assert data is not None
        assert len(data) == 100

    def test_required_columns(self):
        """Generated data should have 'ds' and 'y' columns."""
        dl = DataLayer()
        data = dl.generate_historical_data(50)
        assert 'ds' in data.columns
        assert 'y' in data.columns
        assert 'day_of_week' in data.columns
        assert 'month' in data.columns

    def test_demand_is_positive(self):
        """All demand values should be positive."""
        dl = DataLayer()
        data = dl.generate_historical_data(365)
        assert (data['y'] > 0).all()

    def test_data_source_set(self):
        """data_source should be set to 'synthetic'."""
        dl = DataLayer()
        dl.generate_historical_data(100)
        assert dl.data_source == 'synthetic'

    def test_data_summary(self):
        """get_data_summary should return valid stats."""
        dl = DataLayer()
        dl.generate_historical_data(100)
        summary = dl.get_data_summary()
        assert summary is not None
        assert summary['total_days'] == 100
        assert summary['mean_demand'] > 0


class TestM5Loader:
    def test_fallback_to_synthetic(self):
        """M5 loader should fallback to synthetic when file not found."""
        dl = DataLayer()
        data = dl.load_real_data(csv_path='nonexistent_file.csv')
        assert data is not None
        assert len(data) > 0
        assert dl.data_source == 'synthetic'


class TestSimulationLogs:
    def test_save_log(self):
        """save_simulation_log should store entries."""
        dl = DataLayer()
        dl.save_simulation_log({'day': 1, 'agent': 'test', 'message': 'hello'})
        assert len(dl.simulation_logs) == 1

    def test_get_recent_logs(self):
        """get_recent_logs should return last N entries."""
        dl = DataLayer()
        for i in range(50):
            dl.save_simulation_log({'day': i, 'agent': 'test', 'message': f'event {i}'})
        recent = dl.get_recent_logs(10)
        assert len(recent) == 10
        assert recent[-1]['day'] == 49
