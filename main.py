
"""
USAGE:
1. Install dependencies: pip install -r requirements.txt
2. Set API key: export ANTHROPIC_API_KEY='your-key-here'
3. Run dashboard: streamlit run streamlit_app.py

OR run simulation programmatically:

from data_layer import DataLayer
from forecasting_module import DemandForecaster
from supply_chain_model import SupplyChainModel
from kpi_evaluator import KPIEvaluator

# Initialize
data_layer = DataLayer()
historical_data = data_layer.generate_historical_data(365)

# Train forecaster
forecaster = DemandForecaster(method='lstm')
forecaster.train(historical_data, epochs=50)

# Create model
model = SupplyChainModel(forecaster, data_layer)

# Run simulation
for day in range(30):
    model.step()
    if day == 10:
        model.inject_disruption('supplier', 3)
    print(f"Day {day}: Inventory={model.warehouse.inventory}, Demand={model.daily_demand}")

# Get results
results = model.datacollector.get_model_vars_dataframe()
print(results)
"""

from data_layer import DataLayer
from forecasting_module import DemandForecaster
from supply_chain_model import SupplyChainModel
from kpi_evaluator import KPIEvaluator

# Initialize
data_layer = DataLayer()
historical_data = data_layer.generate_historical_data(365)

# Train forecaster
forecaster = DemandForecaster(method='lstm')
forecaster.train(historical_data, epochs=50)

# Create model
model = SupplyChainModel(forecaster, data_layer)

# Run simulation
for day in range(30):
    model.step()
    if day == 10:
        model.inject_disruption('supplier', 3)
    print(f"Day {day}: Inventory={model.warehouse.inventory}, Demand={model.daily_demand}")

# Get results
results = model.datacollector.get_model_vars_dataframe()
print(results)