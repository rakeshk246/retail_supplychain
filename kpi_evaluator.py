# kpi_evaluator.py
# ==============================================================================
# KPI Evaluator — Supply Chain Performance Metrics
# Phase 1: Enhanced with recovery time tracking, daily snapshots, baseline export
# ==============================================================================

import json
import numpy as np
from datetime import datetime


class KPIEvaluator:
    """Evaluate supply chain performance metrics.
    
    Tracks:
    - Fill Rate: % of demand fulfilled
    - Stockout Rate: % of days with stockouts
    - Average Inventory: mean daily inventory level
    - Resilience Index: ability to maintain service during disruptions
    - Customer Satisfaction: composite score
    - Disruption Recovery Time: days to return to >90% fill rate after disruption
    """

    def __init__(self):
        self.metrics = {
            'total_demand': 0,
            'fulfilled_demand': 0,
            'stockouts': 0,
            'total_days': 0,
            'inventory_sum': 0,
            'orders_placed': 0,
            'orders_delivered': 0
        }

        # Daily snapshots for trend analysis
        self.daily_snapshots = []

        # Disruption recovery tracking
        self.disruption_events = []  # List of {start_day, recovered_day, type}
        self._active_disruption = None
        self._post_disruption_fill_rates = []

    def update(self, demand, fulfilled, inventory, stockout,
               day=None, disruption_active=False):
        """Update metrics with today's data.
        
        Args:
            demand: today's demand
            fulfilled: units actually fulfilled
            inventory: current inventory level
            stockout: whether a stockout occurred today
            day: current simulation day
            disruption_active: whether any disruption is currently active
        """
        self.metrics['total_demand'] += demand
        self.metrics['fulfilled_demand'] += fulfilled
        if stockout:
            self.metrics['stockouts'] += 1
        self.metrics['total_days'] += 1
        self.metrics['inventory_sum'] += inventory

        # Calculate today's fill rate
        today_fill_rate = (fulfilled / max(demand, 1)) * 100

        # Track disruption recovery
        if disruption_active and self._active_disruption is None:
            self._active_disruption = {
                'start_day': day or self.metrics['total_days'],
                'type': 'disruption'
            }
            self._post_disruption_fill_rates = []
        elif not disruption_active and self._active_disruption is not None:
            # Disruption just ended — start tracking recovery
            self._post_disruption_fill_rates.append(today_fill_rate)
            if today_fill_rate >= 90.0:
                # Recovered!
                recovery_day = day or self.metrics['total_days']
                self._active_disruption['recovered_day'] = recovery_day
                self._active_disruption['recovery_time'] = (
                    recovery_day - self._active_disruption['start_day']
                )
                self.disruption_events.append(self._active_disruption)
                self._active_disruption = None
                self._post_disruption_fill_rates = []
        elif self._active_disruption is not None:
            self._post_disruption_fill_rates.append(today_fill_rate)

        # Save daily snapshot
        snapshot = {
            'day': day or self.metrics['total_days'],
            'demand': demand,
            'fulfilled': fulfilled,
            'inventory': inventory,
            'stockout': stockout,
            'fill_rate': today_fill_rate,
            'disruption_active': disruption_active
        }
        self.daily_snapshots.append(snapshot)

    def calculate_kpis(self):
        """Calculate all KPIs."""
        if self.metrics['total_days'] == 0:
            return {}

        fill_rate = (self.metrics['fulfilled_demand'] /
                     max(self.metrics['total_demand'], 1)) * 100

        stockout_rate = (self.metrics['stockouts'] /
                         self.metrics['total_days']) * 100

        avg_inventory = (self.metrics['inventory_sum'] /
                         self.metrics['total_days'])

        resilience_index = max(0, 100 - stockout_rate * 2)

        customer_satisfaction = fill_rate * 0.7 + resilience_index * 0.3

        # Average disruption recovery time
        if self.disruption_events:
            avg_recovery_time = np.mean([
                e['recovery_time'] for e in self.disruption_events
            ])
        else:
            avg_recovery_time = 0

        return {
            'Fill Rate (%)': round(fill_rate, 2),
            'Stock-out Rate (%)': round(stockout_rate, 2),
            'Avg Inventory': round(avg_inventory, 2),
            'Resilience Index': round(resilience_index, 2),
            'Customer Satisfaction': round(customer_satisfaction, 2),
            'Avg Recovery Time (days)': round(avg_recovery_time, 1),
            'Total Disruptions': len(self.disruption_events),
            'Total Days': self.metrics['total_days']
        }

    def get_daily_trend(self, metric='fill_rate'):
        """Get daily trend for a specific metric."""
        if not self.daily_snapshots:
            return []
        return [s.get(metric, 0) for s in self.daily_snapshots]

    def get_rolling_fill_rate(self, window=7):
        """Calculate rolling fill rate over a window."""
        fill_rates = self.get_daily_trend('fill_rate')
        if len(fill_rates) < window:
            return fill_rates

        rolling = []
        for i in range(len(fill_rates) - window + 1):
            rolling.append(np.mean(fill_rates[i:i + window]))
        return rolling

    def export_baseline(self, filename='baseline.json'):
        """Export baseline KPIs to JSON for future comparison.
        
        This is critical: every agentic AI improvement in later phases
        must beat this baseline to prove value.
        """
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'kpis': self.calculate_kpis(),
            'daily_snapshots': self.daily_snapshots,
            'disruption_events': self.disruption_events,
            'raw_metrics': self.metrics
        }

        with open(filename, 'w') as f:
            json.dump(baseline, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("BASELINE KPIs EXPORTED")
        print(f"{'='*60}")
        kpis = self.calculate_kpis()
        for key, value in kpis.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}")
        print(f"Saved to: {filename}")

        return baseline

    def compare_with_baseline(self, baseline_file='baseline.json'):
        """Compare current KPIs with saved baseline."""
        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
        except FileNotFoundError:
            print("No baseline found. Run baseline first.")
            return None

        current = self.calculate_kpis()
        baseline_kpis = baseline['kpis']

        comparison = {}
        for key in current:
            if key in baseline_kpis:
                diff = current[key] - baseline_kpis[key]
                improvement = 'better' if diff > 0 else 'worse' if diff < 0 else 'same'

                # For stockout rate, lower is better
                if 'Stock-out' in key or 'Recovery Time' in key:
                    improvement = 'better' if diff < 0 else 'worse' if diff > 0 else 'same'

                comparison[key] = {
                    'baseline': baseline_kpis[key],
                    'current': current[key],
                    'diff': round(diff, 2),
                    'direction': improvement
                }

        return comparison

    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'total_demand': 0,
            'fulfilled_demand': 0,
            'stockouts': 0,
            'total_days': 0,
            'inventory_sum': 0,
            'orders_placed': 0,
            'orders_delivered': 0
        }
        self.daily_snapshots = []
        self.disruption_events = []
        self._active_disruption = None
        self._post_disruption_fill_rates = []
