# run_agentic.py
# ==============================================================================
# Phase 2: Run Agentic Simulation & Compare with Baseline
# ==============================================================================

import json
import os
from dotenv import load_dotenv

from data_layer import DataLayer
from forecasting_module import DemandForecaster
from agentic_model import AgenticSupplyChainModel
from agentic_agents import LLMEngine
from kpi_evaluator import KPIEvaluator

load_dotenv()


def run_agentic_simulation(days=30, mode='agentic', method='lstm',
                           epochs=30, use_real_data=True):
    """Run simulation with agentic (LLM-powered) agents.
    
    Args:
        days: simulation duration
        mode: 'agentic', 'hybrid', or 'rule_based'
        method: forecasting method
        epochs: LSTM training epochs
        use_real_data: use M5 dataset if available
    """
    print("=" * 60)
    print("AGENTIC AI SUPPLY CHAIN SIMULATION")
    print(f"Phase 2: LLM-Powered Agents | Mode: {mode.upper()}")
    print("=" * 60)

    # Reset LLM engine for fresh stats
    LLMEngine.reset()

    # 1. Initialize data
    data_layer = DataLayer()
    if use_real_data:
        historical_data = data_layer.load_real_data()
    else:
        historical_data = data_layer.generate_historical_data(365)

    data_summary = data_layer.get_data_summary()
    print(f"\nData Source: {data_summary['source']} ({data_summary['total_days']} days)")

    # 2. Train forecaster
    print(f"Training {method.upper()} forecaster...")
    forecaster = DemandForecaster(method=method)
    forecaster.train(historical_data, epochs=epochs)

    # 3. Create agentic model
    model = AgenticSupplyChainModel(forecaster, data_layer, agent_mode=mode)
    kpi_eval = KPIEvaluator()

    # 4. Run simulation
    print(f"\nRunning {days}-day simulation...")
    print("-" * 70)
    print(f"{'Day':>4} | {'Inv':>5} | {'Demand':>6} | {'Supplier':>10} | {'Logistics':>10} | {'Ship':>4} | Agent Decision")
    print("-" * 70)

    for day in range(days):
        model.step()

        # Inject disruptions
        if day == 10:
            model.inject_disruption('supplier', 3)
        if day == 20:
            model.inject_disruption('logistics', 3)

        # Track KPIs
        fulfilled = min(model.daily_demand,
                        model.warehouse.inventory + model.daily_demand)
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

        # Get agent reasoning
        reasoning = model.get_agent_reasoning()
        decision_summary = ""
        if reasoning:
            # Show first reasoning available
            for agent, reason in reasoning.items():
                decision_summary = f"{agent}: {reason[:40]}..."
                break

        status = model.get_status()
        print(
            f"{status['day']:>4} | "
            f"{status['inventory']:>5} | "
            f"{status['demand']:>6} | "
            f"{status['supplier_status']:>10} | "
            f"{status['logistics_status']:>10} | "
            f"{status['pending_shipments']:>4} | "
            f"{decision_summary[:40]}"
        )

    # 5. Results
    print("\n" + "=" * 60)
    print("AGENTIC SIMULATION RESULTS")
    print("=" * 60)

    kpis = kpi_eval.calculate_kpis()
    for key, value in kpis.items():
        print(f"  {key}: {value}")

    # LLM usage stats
    llm = LLMEngine.get_instance()
    llm_stats = llm.get_stats()
    print(f"\nLLM Usage:")
    print(f"  Model: {llm_stats['model']}")
    print(f"  Available: {llm_stats['available']}")
    print(f"  Total API Calls: {llm_stats['total_calls']}")
    print(f"  Total Tokens: {llm_stats['total_tokens']}")

    # 6. Compare with baseline if available
    if os.path.exists('baseline.json'):
        print("\n" + "=" * 60)
        print("BASELINE COMPARISON")
        print("=" * 60)
        comparison = kpi_eval.compare_with_baseline('baseline.json')
        if comparison:
            for key, data in comparison.items():
                arrow = "↑" if data['direction'] == 'better' else "↓" if data['direction'] == 'worse' else "="
                print(f"  {arrow} {key}: {data['baseline']} → {data['current']} ({data['diff']:+.2f})")
        else:
            print("  Run 'python run_baseline.py' first to generate baseline.")

    # 7. Save agentic results
    agentic_results = {
        'mode': mode,
        'kpis': kpis,
        'llm_stats': llm_stats,
        'days': days,
        'data_source': data_summary['source']
    }
    with open('agentic_results.json', 'w') as f:
        json.dump(agentic_results, f, indent=2)
    print(f"\nResults saved to agentic_results.json")

    return model, kpi_eval


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'agentic'
    run_agentic_simulation(days=30, mode=mode)
