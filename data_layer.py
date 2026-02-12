import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class DataLayer:
    """Manages historical data collection and storage"""
    
    def __init__(self):
        self.historical_data = None
        self.simulation_logs = []
        
    def generate_historical_data(self, days=365):
        """Generate synthetic historical sales data"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic demand patterns
        baseline = 100
        trend = np.linspace(0, 50, days)
        seasonal = 30 * np.sin(np.arange(days) * 2 * np.pi / 7)  # Weekly pattern
        noise = np.random.normal(0, 15, days)
        
        demand = baseline + trend + seasonal + noise
        demand = np.maximum(demand, 0)
        
        self.historical_data = pd.DataFrame({
            'ds': dates,
            'y': demand,
            'day_of_week': dates.dayofweek,
            'month': dates.month
        })
        
        return self.historical_data
    
    def save_simulation_log(self, log_entry):
        """Save simulation event logs"""
        self.simulation_logs.append({
            'timestamp': datetime.now(),
            **log_entry
        })
    
    def export_logs(self, filename='simulation_logs.json'):
        """Export logs to file"""
        with open(filename, 'w') as f:
            json.dump(self.simulation_logs, f, indent=2, default=str)
