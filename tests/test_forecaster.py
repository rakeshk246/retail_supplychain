# tests/test_forecaster.py
# ==============================================================================
# Tests for DemandForecaster — LSTM, Prophet, predict_next
# ==============================================================================

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer import DataLayer
from forecasting_module import DemandForecaster


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    dl = DataLayer()
    return dl.generate_historical_data(100)


class TestDemandForecaster:
    def test_train_lstm(self, sample_data):
        """LSTM should train without errors."""
        forecaster = DemandForecaster(method='lstm')
        forecaster.train(sample_data, epochs=3)
        assert forecaster.is_trained
        assert forecaster.data is not None

    def test_predict_next_bug2_regression(self, sample_data):
        """BUG 2 REGRESSION: predict_next() should not raise IndentationError."""
        forecaster = DemandForecaster(method='lstm')
        forecaster.train(sample_data, epochs=3)
        # This would have raised IndentationError before the fix
        result = forecaster.predict_next()
        assert isinstance(result, float)
        assert result > 0

    def test_predict_lstm(self, sample_data):
        """LSTM prediction should return array of correct length."""
        forecaster = DemandForecaster(method='lstm')
        forecaster.train(sample_data, epochs=3)
        recent = sample_data.tail(30)
        result = forecaster.predict(recent, steps=3)
        assert len(result) == 3
        assert all(isinstance(v, (float, np.floating)) for v in result)

    def test_predict_without_training(self, sample_data):
        """Prediction without training should use fallback."""
        forecaster = DemandForecaster(method='lstm')
        forecaster.data = sample_data
        result = forecaster.predict_next()
        assert result > 0

    def test_confidence_score(self, sample_data):
        """Confidence score should be between 0 and 1."""
        forecaster = DemandForecaster(method='lstm')
        forecaster.train(sample_data, epochs=3)
        confidence = forecaster.get_confidence()
        assert 0 <= confidence <= 1

    def test_rmse_calculation(self):
        """RMSE calculation should work correctly."""
        forecaster = DemandForecaster()
        actual = np.array([100, 110, 90, 105])
        predicted = np.array([98, 112, 88, 107])
        rmse = forecaster.get_rmse(actual, predicted)
        assert rmse > 0
        assert rmse < 10  # Should be small for close values
