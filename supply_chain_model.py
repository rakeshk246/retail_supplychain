from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from agent import SupplierAgent, WarehouseAgent, LogisticsAgent, DemandAgent
from mesa.datacollection import DataCollector


class SupplyChainModel(Model):
    """Mesa model for supply chain simulation"""
    
    def __init__(self, forecaster, data_layer):
        super().__init__()
        
        self.forecaster = forecaster
        self.data_layer = data_layer
        self.schedule = RandomActivation(self)
        self.current_day = 0
        self.daily_demand = 0
        self.stockouts = 0
        
        # Create agents
        self.supplier = SupplierAgent(1, self)
        self.warehouse = WarehouseAgent(2, self)
        self.logistics = LogisticsAgent(3, self)
        self.demand_agent = DemandAgent(4, self, forecaster)
        
        # Add agents to schedule
        for agent in [self.supplier, self.warehouse, self.logistics, self.demand_agent]:
            self.schedule.add(agent)
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Inventory": lambda m: m.warehouse.inventory,
                "Demand": lambda m: m.daily_demand,
                "Stockouts": lambda m: m.stockouts
            }
        )
        
        self.event_log = []
    
    def log_event(self, agent_type, message):
        """Log simulation events"""
        self.event_log.append({
            'day': self.current_day,
            'agent': agent_type,
            'message': message
        })
        self.data_layer.save_simulation_log({
            'day': self.current_day,
            'agent': agent_type,
            'message': message
        })
    
    def get_agent_by_type(self, agent_class):
        """Get agent by class type"""
        for agent in self.schedule.agents:
            if isinstance(agent, agent_class):
                return agent
        return None
    
    def inject_disruption(self, agent_type, duration=3):
        """Inject disruption for resilience testing"""
        if agent_type == 'supplier':
            self.supplier.status = 'disrupted'
            self.log_event('disruption', f"Supplier disrupted for {duration} days")
        elif agent_type == 'logistics':
            self.logistics.status = 'disrupted'
            self.log_event('disruption', f"Logistics disrupted for {duration} days")
        
        return duration
    
    def step(self):
        """Execute one simulation step"""
        self.current_day += 1

        # 1. Demand prediction
        self.demand_agent.step()

        # 2. Fulfill demand BEFORE logistics changes inventory
        fulfilled = self.warehouse.fulfill_demand(self.daily_demand)

        # 3. Predict future demand for reorder decision
        recent_data = self.data_layer.historical_data.tail(30)
        predicted = self.forecaster.predict(recent_data, steps=3)[0]

        # 4. Reorder if needed
        order_qty = self.warehouse.check_reorder(predicted)
        if order_qty > 0:
            order = self.supplier.process_order(order_qty)
            if order:
                self.logistics.schedule_shipment(order)

        # 5. Run all agents (this includes LogisticsAgent.step())
        # Collect data
        self.datacollector.collect(self)
        
        # Run scheduled agents
        self.schedule.step()