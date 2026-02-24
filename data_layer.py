# data_layer.py
# ==============================================================================
# Data Layer — Manages historical data, real dataset loading, and event logging
# Phase 1: Enhanced synthetic data + M5 Walmart dataset support
# ==============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os


class DataLayer:
    """Manages historical data collection and storage.
    
    Supports:
    - Enhanced synthetic data with realistic seasonality
    - M5 Walmart Forecasting Competition dataset (real retail data)
    - Simulation event logging
    """

    def __init__(self):
        self.historical_data = None
        self.simulation_logs = []
        self.data_source = None  # 'synthetic' or 'm5'

    def generate_historical_data(self, days=365):
        """Generate enhanced synthetic historical sales data.
        
        Includes: baseline trend, weekly seasonality, monthly seasonality,
        holiday effects, and random noise for realistic demand patterns.
        """
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Base demand
        baseline = 100

        # Upward trend
        trend = np.linspace(0, 50, days)

        # Weekly seasonality (weekends dip)
        weekly = 30 * np.sin(np.arange(days) * 2 * np.pi / 7)

        # Monthly seasonality (month-end spike)
        monthly = 20 * np.sin(np.arange(days) * 2 * np.pi / 30)

        # Holiday / promotion spikes (random days get 50-100% boost)
        holidays = np.zeros(days)
        holiday_days = np.random.choice(days, size=int(days * 0.05), replace=False)
        holidays[holiday_days] = np.random.uniform(50, 100, len(holiday_days))

        # Random noise
        noise = np.random.normal(0, 15, days)

        # Combine all components
        demand = baseline + trend + weekly + monthly + holidays + noise
        demand = np.maximum(demand, 1)  # No zero or negative demand

        self.historical_data = pd.DataFrame({
            'ds': dates,
            'y': demand,
            'day_of_week': dates.dayofweek,
            'month': dates.month
        })

        self.data_source = 'synthetic'
        return self.historical_data

    def load_real_data(self, csv_path=None, store_id='CA_1', item_id='FOODS_3_090'):
        """Load Walmart M5 dataset — real retail demand data.
        
        Args:
            csv_path: path to sales_train_evaluation.csv
            store_id: Walmart store ID (e.g., 'CA_1')
            item_id: Product item ID (e.g., 'FOODS_3_090')
            
        Returns:
            DataFrame with 'ds' (date) and 'y' (demand) columns
        """
        if csv_path is None:
            # Try default locations
            possible_paths = [
                'data/sales_train_evaluation.csv',
                'sales_train_evaluation.csv',
                '../data/sales_train_evaluation.csv'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break

        if csv_path is None or not os.path.exists(csv_path):
            print("M5 dataset not found. Using enhanced synthetic data instead.")
            return self.generate_historical_data(365 * 3)  # 3 years synthetic

        try:
            df = pd.read_csv(csv_path)

            # Filter for specific store and item
            item_row = df[(df['store_id'] == store_id) & (df['item_id'] == item_id)]

            if item_row.empty:
                print(f"Item {item_id} at store {store_id} not found. Using first available item.")
                item_row = df.iloc[[0]]

            # Reshape from wide to long format
            day_cols = [c for c in df.columns if c.startswith('d_')]
            demand_series = item_row[day_cols].values.flatten()

            # M5 dataset starts from 2011-01-29
            dates = pd.date_range(start='2011-01-29', periods=len(demand_series), freq='D')

            self.historical_data = pd.DataFrame({
                'ds': dates,
                'y': demand_series.astype(float),
                'day_of_week': dates.dayofweek,
                'month': dates.month
            })

            # Remove any days with zero demand (store closed)
            self.historical_data.loc[self.historical_data['y'] == 0, 'y'] = 1

            self.data_source = 'm5'
            print(f"Loaded M5 data: {len(self.historical_data)} days for {item_id} at {store_id}")
            return self.historical_data

        except Exception as e:
            print(f"Error loading M5 data: {e}. Using synthetic data.")
            return self.generate_historical_data(365 * 3)

    def get_data_summary(self):
        """Get summary statistics of the loaded data."""
        if self.historical_data is None:
            return None

        return {
            'source': self.data_source,
            'total_days': len(self.historical_data),
            'mean_demand': round(self.historical_data['y'].mean(), 2),
            'std_demand': round(self.historical_data['y'].std(), 2),
            'min_demand': round(self.historical_data['y'].min(), 2),
            'max_demand': round(self.historical_data['y'].max(), 2),
            'date_range': f"{self.historical_data['ds'].min()} to {self.historical_data['ds'].max()}"
        }

    def save_simulation_log(self, log_entry):
        """Save simulation event logs."""
        self.simulation_logs.append({
            'timestamp': datetime.now().isoformat(),
            **log_entry
        })

    def export_logs(self, filename='simulation_logs.json'):
        """Export logs to file."""
        with open(filename, 'w') as f:
            json.dump(self.simulation_logs, f, indent=2, default=str)
        print(f"Exported {len(self.simulation_logs)} log entries to {filename}")

    def get_recent_logs(self, n=20):
        """Get the most recent n log entries."""
        return self.simulation_logs[-n:]
