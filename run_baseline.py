# run_baseline.py
# ==============================================================================
# Baseline Runner — Run 90-day simulation and save performance baseline
# This baseline is the standard against which all agentic AI improvements
# will be measured in Phases 2-5.
# ==============================================================================

from data_layer import DataLayer
from forecasting_module import DemandForecaster
from supply_chain_model import SupplyChainModel
from kpi_evaluator import KPIEvaluator


def run_baseline(days=90, method='lstm', epochs=50, runs=3):
    """Run baseline simulation and export KPIs.
    
    Args:
        days: simulation duration (default 90)
        method: forecasting method ('lstm' or 'prophet')
        epochs: LSTM training epochs
        runs: number of runs to average (for statistical significance)
    """
    print("=" * 60)
    print("BASELINE KPI COLLECTION")
    print(f"Configuration: {days} days × {runs} runs")
    print(f"Forecaster: {method.upper()}")
    print("=" * 60)

    all_kpis = []

    for run_num in range(1, runs + 1):
        print(f"\n--- Run {run_num}/{runs} ---")

        # Initialize
        data_layer = DataLayer()
        historical_data = data_layer.load_real_data()

        forecaster = DemandForecaster(method=method)
        forecaster.train(historical_data, epochs=epochs)

        model = SupplyChainModel(forecaster, data_layer)
        kpi_eval = KPIEvaluator()

        # Run simulation with disruptions at day 30 and 60
        for day in range(days):
            model.step()

            # Inject disruptions at standard test points
            if day == 30:
                model.inject_disruption('supplier', duration=5)
            if day == 60:
                model.inject_disruption('logistics', duration=4)

            # Track KPIs
            fulfilled = min(model.daily_demand, model.warehouse.inventory + model.daily_demand)
            stockout = model.warehouse.inventory <= 0
            disruption_active = bool(model.disruption_schedule)

            kpi_eval.update(
                demand=model.daily_demand,
                fulfilled=fulfilled,
                inventory=model.warehouse.inventory,
                stockout=stockout,
                day=model.current_day,
                disruption_active=disruption_active
            )

        run_kpis = kpi_eval.calculate_kpis()
        all_kpis.append(run_kpis)

        print(f"  Fill Rate: {run_kpis['Fill Rate (%)']:.1f}%")
        print(f"  Stockout Rate: {run_kpis['Stock-out Rate (%)']:.1f}%")
        print(f"  Avg Inventory: {run_kpis['Avg Inventory']:.0f}")

    # Export the last run's detailed baseline (with daily snapshots)
    kpi_eval.export_baseline('baseline.json')

    # Print averaged results
    print("\n" + "=" * 60)
    print("AVERAGED BASELINE RESULTS")
    print("=" * 60)

    import numpy as np
    for key in all_kpis[0]:
        values = [k[key] for k in all_kpis]
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 0
        print(f"  {key}: {mean:.2f} ± {std:.2f}")

    print("\n✅ Baseline saved to baseline.json")
    print("This baseline will be used to evaluate agentic AI improvements.")

    return all_kpis


if __name__ == "__main__":
    run_baseline()
