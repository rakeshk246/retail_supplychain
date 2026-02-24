# supply_chain_model.py
# ==============================================================================
# Mesa Supply Chain Model — Phase 1 (Foundation)
# Compatible with Mesa 3.0+ (Agent takes only model, unique_id auto-assigned)
# Bug fixes: Import path, disruption auto-recovery, proper logistics integration
# ==============================================================================

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# BUG 1 FIX: Import from 'agents' (not 'agent')
from agents import SupplierAgent, WarehouseAgent, LogisticsAgent, DemandAgent


class SupplyChainModel(Model):
    """Mesa model for supply chain simulation.
    
    Orchestrates all agents through a daily step cycle:
    1. Demand prediction
    2. Demand fulfillment
    3. Reorder check
    4. Supplier order processing
    5. Logistics scheduling & delivery
    """

    def __init__(self, forecaster, data_layer,
                 initial_inventory=500, reorder_point=200, reorder_qty=300):
        super().__init__()

        self.forecaster = forecaster
        self.data_layer = data_layer
        self.schedule = RandomActivation(self)
        self.current_day = 0
        self.daily_demand = 0
        self.stockouts = 0

        # BUG 3 FIX: Disruption auto-recovery schedule
        self.disruption_schedule = {}  # {'supplier': end_day, 'logistics': end_day}

        # Create agents (Mesa 3.0+ — no unique_id needed)
        self.supplier = SupplierAgent(self)
        self.warehouse = WarehouseAgent(
            self,
            initial_inventory=initial_inventory,
            reorder_point=reorder_point,
            reorder_qty=reorder_qty
        )
        self.logistics = LogisticsAgent(self)
        self.demand_agent = DemandAgent(self, forecaster)

        # Add agents to schedule
        for agent in [self.supplier, self.warehouse, self.logistics, self.demand_agent]:
            self.schedule.add(agent)

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Inventory": lambda m: m.warehouse.inventory,
                "Demand": lambda m: m.daily_demand,
                "Stockouts": lambda m: m.stockouts,
                "Supplier_Status": lambda m: m.supplier.status,
                "Logistics_Status": lambda m: m.logistics.status,
                "Pending_Shipments": lambda m: len(m.logistics.shipments),
            }
        )

        self.event_log = []

    def log_event(self, agent_type, message):
        """Log simulation events."""
        event = {
            'day': self.current_day,
            'agent': agent_type,
            'message': message
        }
        self.event_log.append(event)
        self.data_layer.save_simulation_log(event)

    def get_agent_by_type(self, agent_class):
        """Get agent by class type."""
        for agent in self.schedule.agents:
            if isinstance(agent, agent_class):
                return agent
        return None

    # BUG 3 FIX: Disruption injection with auto-recovery scheduling
    def inject_disruption(self, agent_type, duration=3):
        """Inject disruption for resilience testing with auto-recovery.
        
        Args:
            agent_type: 'supplier' or 'logistics'
            duration: number of days the disruption lasts
        """
        end_day = self.current_day + duration
        self.disruption_schedule[agent_type] = end_day

        if agent_type == 'supplier':
            self.supplier.status = 'disrupted'
            self.log_event('Disruption',
                           f"Supplier DISRUPTED for {duration} days (recovers day {end_day})")
        elif agent_type == 'logistics':
            self.logistics.status = 'disrupted'
            self.log_event('Disruption',
                           f"Logistics DISRUPTED for {duration} days (recovers day {end_day})")

        return duration

    def _check_disruption_recovery(self):
        """BUG 3 FIX: Auto-recover disrupted agents when duration expires."""
        recovered = []
        for agent_type, end_day in list(self.disruption_schedule.items()):
            if self.current_day >= end_day:
                if agent_type == 'supplier' and self.supplier.status == 'disrupted':
                    self.supplier.status = 'active'
                    self.log_event('Recovery', 'Supplier recovered from disruption')
                    recovered.append(agent_type)
                elif agent_type == 'logistics' and self.logistics.status == 'disrupted':
                    self.logistics.status = 'active'
                    self.log_event('Recovery', 'Logistics recovered from disruption')
                    recovered.append(agent_type)

        for agent_type in recovered:
            del self.disruption_schedule[agent_type]

    def step(self):
        """Execute one simulation day.
        
        Order of operations:
        1. Advance day counter
        2. Check disruption recoveries
        3. Generate/predict demand
        4. Fulfill demand from warehouse
        5. Check reorder point
        6. Place supplier order if needed
        7. Schedule logistics shipment
        8. Process logistics deliveries (arrived shipments)
        9. Collect data
        """
        self.current_day += 1

        # BUG 3 FIX: Auto-recovery check at start of each day
        self._check_disruption_recovery()

        # 1. Demand prediction
        self.demand_agent.step()

        # 2. Fulfill demand BEFORE logistics changes inventory
        fulfilled = self.warehouse.fulfill_demand(self.daily_demand)

        # 3. Predict future demand for reorder decision
        try:
            recent_data = self.data_layer.historical_data.tail(30)
            predicted = self.forecaster.predict(recent_data, steps=3)[0]
        except Exception:
            predicted = self.daily_demand  # Fallback

        # 4. Reorder if needed
        order_qty = self.warehouse.check_reorder(predicted)
        if order_qty > 0:
            order = self.supplier.process_order(order_qty)
            if order and isinstance(order, dict):
                # BUG 4 FIX: Pass order dict to logistics (handles lead time)
                self.logistics.schedule_shipment(order)
            elif order and isinstance(order, (int, float)) and order > 0:
                self.logistics.schedule_shipment(int(order))

        # 5. Process deliveries (arrived shipments)
        self.logistics.step()

        # 6. Collect data
        self.datacollector.collect(self)

    def get_status(self):
        """Get current model status summary."""
        return {
            'day': self.current_day,
            'inventory': self.warehouse.inventory,
            'demand': self.daily_demand,
            'stockouts': self.stockouts,
            'supplier_status': self.supplier.status,
            'logistics_status': self.logistics.status,
            'pending_shipments': len(self.logistics.shipments),
            'active_disruptions': dict(self.disruption_schedule)
        }