# agentic_model.py
# ==============================================================================
# Phase 2+3: Agentic Supply Chain Model
# Supports rule-based, agentic, and hybrid modes
# Phase 3: Memory + Inter-Agent Communication
# ==============================================================================

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from agents import SupplierAgent, WarehouseAgent, LogisticsAgent, DemandAgent
from agentic_agents import (
    AgenticSupplierAgent, AgenticWarehouseAgent,
    AgenticLogisticsAgent, AgenticDemandAgent,
    LLMEngine
)
from message_bus import MessageBus


class AgenticSupplyChainModel(Model):
    """Supply chain model supporting both rule-based and agentic modes.
    
    Modes:
    - 'rule_based': Original Phase 1 agents (no LLM calls)
    - 'agentic': LLM-powered ReAct agents via Groq
    - 'hybrid': LLM for warehouse/supplier decisions, rule-based for others
    """

    def __init__(self, forecaster, data_layer,
                 agent_mode='agentic',
                 initial_inventory=500, reorder_point=200, reorder_qty=300):
        super().__init__()

        self.forecaster = forecaster
        self.data_layer = data_layer
        self.agent_mode = agent_mode
        self.schedule = RandomActivation(self)
        self.current_day = 0
        self.daily_demand = 0
        self.stockouts = 0

        # Disruption auto-recovery schedule
        self.disruption_schedule = {}

        # Phase 3: Initialize message bus
        MessageBus.reset()
        self.bus = MessageBus.get_instance()

        # Create agents based on mode
        if agent_mode == 'agentic':
            self.supplier = AgenticSupplierAgent(self)
            self.warehouse = AgenticWarehouseAgent(
                self,
                initial_inventory=initial_inventory,
                reorder_point=reorder_point,
                reorder_qty=reorder_qty
            )
            self.logistics = AgenticLogisticsAgent(self)
            self.demand_agent = AgenticDemandAgent(self, forecaster)
            print(f"Mode: AGENTIC (LLM-powered agents)")
        elif agent_mode == 'hybrid':
            self.supplier = AgenticSupplierAgent(self)
            self.warehouse = AgenticWarehouseAgent(
                self,
                initial_inventory=initial_inventory,
                reorder_point=reorder_point,
                reorder_qty=reorder_qty
            )
            self.logistics = LogisticsAgent(self)
            self.demand_agent = DemandAgent(self, forecaster)
            print(f"Mode: HYBRID (AI supplier/warehouse, rule-based logistics/demand)")
        else:
            self.supplier = SupplierAgent(self)
            self.warehouse = WarehouseAgent(
                self,
                initial_inventory=initial_inventory,
                reorder_point=reorder_point,
                reorder_qty=reorder_qty
            )
            self.logistics = LogisticsAgent(self)
            self.demand_agent = DemandAgent(self, forecaster)
            print(f"Mode: RULE-BASED (no LLM calls)")

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
                "Agent_Mode": lambda m: m.agent_mode,
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

    def inject_disruption(self, agent_type, duration=3):
        """Inject disruption with auto-recovery."""
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
        """Auto-recover disrupted agents."""
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
        """Execute one simulation day."""
        self.current_day += 1
        self._check_disruption_recovery()

        # 1. Demand
        self.demand_agent.step()

        # 2. Fulfill demand
        fulfilled = self.warehouse.fulfill_demand(self.daily_demand)

        # 3. Predict future demand
        try:
            recent_data = self.data_layer.historical_data.tail(30)
            predicted = self.forecaster.predict(recent_data, steps=3)[0]
        except Exception:
            predicted = self.daily_demand

        # 4. Reorder if needed
        order_qty = self.warehouse.check_reorder(predicted)
        if order_qty > 0:
            order = self.supplier.process_order(order_qty)
            if order and isinstance(order, dict):
                self.logistics.schedule_shipment(order)
            elif order and isinstance(order, (int, float)) and order > 0:
                self.logistics.schedule_shipment(int(order))

        # 5. Process deliveries
        self.logistics.step()

        # 6. Collect data
        self.datacollector.collect(self)

    def get_status(self):
        """Get model status."""
        status = {
            'day': self.current_day,
            'mode': self.agent_mode,
            'inventory': self.warehouse.inventory,
            'demand': self.daily_demand,
            'stockouts': self.stockouts,
            'supplier_status': self.supplier.status,
            'logistics_status': self.logistics.status,
            'pending_shipments': len(self.logistics.shipments),
            'active_disruptions': dict(self.disruption_schedule)
        }
        
        # Add LLM stats
        try:
            llm = LLMEngine.get_instance()
            status['llm_stats'] = llm.get_stats()
        except Exception:
            pass
        
        # Add message bus stats
        try:
            status['bus_stats'] = self.bus.get_stats()
        except Exception:
            pass
        
        return status

    def get_agent_reasoning(self):
        """Get the last reasoning from each agentic agent."""
        reasoning = {}
        for agent_name, agent in [
            ('Supplier', self.supplier),
            ('Warehouse', self.warehouse),
            ('Logistics', self.logistics),
            ('Demand', self.demand_agent)
        ]:
            if hasattr(agent, 'last_reasoning') and agent.last_reasoning:
                reasoning[agent_name] = agent.last_reasoning
        return reasoning

    def get_memory_stats(self):
        """Get memory statistics from all agents."""
        stats = {}
        for agent_name, agent in [
            ('Supplier', self.supplier),
            ('Warehouse', self.warehouse),
            ('Logistics', self.logistics),
            ('Demand', self.demand_agent)
        ]:
            if hasattr(agent, 'memory'):
                stats[agent_name] = agent.memory.get_stats()
        return stats

    def get_message_log(self, n=20):
        """Get recent inter-agent messages."""
        return self.bus.get_message_log(n)
