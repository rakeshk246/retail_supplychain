                # orchestrator.py
# ==============================================================================
# Phase 4: LangGraph Workflow Orchestration
# Adaptive state machine — routes decisions based on supply chain conditions
# ==============================================================================

from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END

from data_layer import DataLayer
from forecasting_module import DemandForecaster
from agentic_agents import (
    AgenticSupplierAgent, AgenticWarehouseAgent,
    AgenticLogisticsAgent, AgenticDemandAgent,
    LLMEngine
)
from agents import SupplierAgent, WarehouseAgent, LogisticsAgent, DemandAgent
from message_bus import MessageBus
from explainability import ExplainabilityEngine, DecisionRecord

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


# ==============================================================================
# State Definition
# ==============================================================================

class SupplyChainState(TypedDict):
    """Full state passed through the LangGraph workflow."""
    day: int
    inventory: int
    demand: int
    predicted_demand: float
    supplier_status: str
    logistics_status: str
    pending_shipments: int
    stockouts: int
    order_placed: bool
    order_result: dict
    workflow_path: str      # tracks which route was taken
    decisions_made: list    # XAI decision IDs for this step


# ==============================================================================
# Orchestrated Supply Chain Model
# ==============================================================================

class OrchestratedSupplyChainModel(Model):
    """LangGraph-orchestrated supply chain with XAI.
    
    Instead of fixed agent ordering, uses a state machine:
    - NORMAL path: standard reorder check
    - EMERGENCY path: low inventory triggers fast ordering
    - CRISIS path: stockout or dual-disruption triggers emergency response
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
        self.disruption_schedule = {}
        self.event_log = []

        # Phase 3: Message bus
        MessageBus.reset()
        self.bus = MessageBus.get_instance()

        # Phase 4: Explainability
        ExplainabilityEngine.reset()
        self.xai = ExplainabilityEngine.get_instance()

        # Create agents based on mode
        if agent_mode in ('agentic', 'hybrid'):
            self.supplier = AgenticSupplierAgent(self)
            self.warehouse = AgenticWarehouseAgent(
                self, initial_inventory=initial_inventory,
                reorder_point=reorder_point, reorder_qty=reorder_qty
            )
            if agent_mode == 'agentic':
                self.logistics = AgenticLogisticsAgent(self)
                self.demand_agent = AgenticDemandAgent(self, forecaster)
            else:
                self.logistics = LogisticsAgent(self)
                self.demand_agent = DemandAgent(self, forecaster)
        else:
            self.supplier = SupplierAgent(self)
            self.warehouse = WarehouseAgent(
                self, initial_inventory=initial_inventory,
                reorder_point=reorder_point, reorder_qty=reorder_qty
            )
            self.logistics = LogisticsAgent(self)
            self.demand_agent = DemandAgent(self, forecaster)

        for agent in [self.supplier, self.warehouse, self.logistics, self.demand_agent]:
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            model_reporters={
                "Inventory": lambda m: m.warehouse.inventory,
                "Demand": lambda m: m.daily_demand,
                "Stockouts": lambda m: m.stockouts,
                "Agent_Mode": lambda m: m.agent_mode,
                "Workflow_Path": lambda m: m._last_workflow_path,
            }
        )

        self._last_workflow_path = "none"
        self._workflow = self._build_workflow()

    # =========================================================================
    # LangGraph Workflow Definition
    # =========================================================================

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(SupplyChainState)

        # Add nodes
        workflow.add_node("forecast_demand", self._node_forecast_demand)
        workflow.add_node("check_inventory", self._node_check_inventory)
        workflow.add_node("normal_reorder", self._node_normal_reorder)
        workflow.add_node("emergency_reorder", self._node_emergency_reorder)
        workflow.add_node("crisis_response", self._node_crisis_response)
        workflow.add_node("supplier_decision", self._node_supplier_decision)
        workflow.add_node("logistics_scheduling", self._node_logistics_scheduling)
        workflow.add_node("process_deliveries", self._node_process_deliveries)
        workflow.add_node("collect_metrics", self._node_collect_metrics)

        # Set entry point
        workflow.set_entry_point("forecast_demand")

        # Edges
        workflow.add_edge("forecast_demand", "check_inventory")

        # Conditional routing based on inventory level
        workflow.add_conditional_edges(
            "check_inventory",
            self._route_by_inventory,
            {
                "normal": "normal_reorder",
                "emergency": "emergency_reorder",
                "crisis": "crisis_response"
            }
        )

        workflow.add_edge("normal_reorder", "supplier_decision")
        workflow.add_edge("emergency_reorder", "supplier_decision")
        workflow.add_edge("crisis_response", "supplier_decision")

        # Conditional: skip logistics if no order was placed
        workflow.add_conditional_edges(
            "supplier_decision",
            self._route_after_supplier,
            {
                "schedule": "logistics_scheduling",
                "skip": "process_deliveries"
            }
        )

        workflow.add_edge("logistics_scheduling", "process_deliveries")
        workflow.add_edge("process_deliveries", "collect_metrics")
        workflow.add_edge("collect_metrics", END)

        return workflow.compile()

    # =========================================================================
    # Routing Functions (Conditional Edges)
    # =========================================================================

    def _route_by_inventory(self, state: SupplyChainState) -> str:
        """Route based on inventory level and disruption status."""
        inv = state['inventory']
        demand = state['demand']
        supplier_down = state['supplier_status'] == 'disrupted'
        logistics_down = state['logistics_status'] == 'disrupted'

        # CRISIS: stockout or near-zero with disruptions
        if inv <= 0 or (inv < demand and (supplier_down or logistics_down)):
            return "crisis"

        # EMERGENCY: below reorder point
        if inv < self.warehouse.reorder_point:
            return "emergency"

        # NORMAL
        return "normal"

    def _route_after_supplier(self, state: SupplyChainState) -> str:
        """Skip logistics if no order was placed."""
        if state['order_placed'] and state['order_result']:
            return "schedule"
        return "skip"

    # =========================================================================
    # Workflow Nodes (each wraps agent logic + XAI)
    # =========================================================================

    def _node_forecast_demand(self, state: SupplyChainState) -> dict:
        """Node 1: Generate demand forecast."""
        self.demand_agent.step()

        # XAI record
        rec = self.xai.create_record("Demand", "forecast", self.current_day)
        rec.action = f"Forecast demand = {self.daily_demand}"
        rec.is_llm_decision = hasattr(self.demand_agent, 'last_reasoning') and self.demand_agent.last_reasoning is not None
        rec.confidence = 0.8

        # WHY
        recent = getattr(self.demand_agent, 'recent_demands', [])[-7:]
        avg = sum(recent) / len(recent) if recent else self.daily_demand
        factors = [
            f"Base model predicted ~{self.daily_demand} units",
            f"Recent 7-day average: {avg:.0f}",
            f"Current inventory: {self.warehouse.inventory}"
        ]
        
        reasoning = getattr(self.demand_agent, 'last_reasoning', '') or ''
        if reasoning:
            rec.set_why(
                reasoning=reasoning,
                summary=f"LLM analyzed demand patterns and forecast {self.daily_demand} units",
                factors=factors
            )
        else:
            rec.set_why(
                reasoning="Used statistical model prediction with random variance",
                summary=f"Statistical model forecast {self.daily_demand} units based on historical patterns",
                factors=factors
            )
            rec.fallback_reason = "LLM unavailable or not in agentic mode"

        return {
            'demand': self.daily_demand,
            'predicted_demand': self.daily_demand,
            'decisions_made': state.get('decisions_made', []) + [len(self.xai.decisions) - 1]
        }

    def _node_check_inventory(self, state: SupplyChainState) -> dict:
        """Node 2: Fulfill demand and check inventory level."""
        fulfilled = self.warehouse.fulfill_demand(self.daily_demand)

        rec = self.xai.create_record("Warehouse", "fulfill_demand", self.current_day)
        rec.action = f"Fulfilled {fulfilled}/{self.daily_demand} units"
        rec.confidence = 1.0
        rec.is_llm_decision = False
        rec.context = {
            'inventory_before': self.warehouse.inventory + fulfilled,
            'inventory_after': self.warehouse.inventory,
            'demand': self.daily_demand
        }

        if fulfilled < self.daily_demand:
            rec.set_why(
                reasoning=f"Only {fulfilled} units available, {self.daily_demand - fulfilled} unfulfilled",
                summary=f"STOCKOUT: insufficient inventory to meet demand ({fulfilled}/{self.daily_demand})",
                factors=[
                    f"Inventory was {self.warehouse.inventory + fulfilled} but demand was {self.daily_demand}",
                    "Customer orders partially unfulfilled"
                ]
            )
        else:
            rec.set_why(
                reasoning="Sufficient inventory to meet demand",
                summary=f"Fully met demand of {self.daily_demand} units, {self.warehouse.inventory} remaining",
                factors=[f"Inventory ({self.warehouse.inventory + fulfilled}) > demand ({self.daily_demand})"]
            )

        return {
            'inventory': self.warehouse.inventory,
            'stockouts': self.stockouts,
            'supplier_status': self.supplier.status,
            'logistics_status': self.logistics.status,
            'pending_shipments': len(self.logistics.shipments)
        }

    def _node_normal_reorder(self, state: SupplyChainState) -> dict:
        """Node 3a: Normal path — standard reorder check."""
        predicted = state.get('predicted_demand', self.daily_demand)
        order_qty = self.warehouse.check_reorder(predicted)

        rec = self.xai.create_record("Warehouse", "reorder_check", self.current_day)
        rec.workflow_path = "NORMAL"
        rec.is_llm_decision = hasattr(self.warehouse, 'last_reasoning') and self.warehouse.last_reasoning is not None
        rec.confidence = 0.7

        if order_qty > 0:
            rec.action = f"Reorder {order_qty} units"
            reasoning = getattr(self.warehouse, 'last_reasoning', '') or ''
            rec.set_why(
                reasoning=reasoning or f"Inventory ({self.warehouse.inventory}) below reorder point ({self.warehouse.reorder_point})",
                summary=f"Normal reorder: inventory at {self.warehouse.inventory} triggered standard replenishment of {order_qty} units",
                factors=[
                    f"Inventory ({self.warehouse.inventory}) < reorder point ({self.warehouse.reorder_point})",
                    f"Predicted demand: {predicted:.0f}",
                    f"Default order quantity: {self.warehouse.reorder_qty}"
                ],
                alternatives=[
                    {'option': 'Hold (no order)',
                     'rejected_because': f'inventory below reorder point of {self.warehouse.reorder_point}'},
                    {'option': 'Emergency order (larger quantity)',
                     'rejected_because': 'inventory not critically low; normal reorder sufficient'}
                ]
            )
        else:
            rec.action = "Hold — no reorder needed"
            rec.set_why(
                reasoning="Inventory above reorder threshold",
                summary=f"No reorder needed: inventory ({self.warehouse.inventory}) is above reorder point ({self.warehouse.reorder_point})",
                factors=[
                    f"Inventory ({self.warehouse.inventory}) >= reorder point ({self.warehouse.reorder_point})"
                ]
            )

        return {
            'order_placed': order_qty > 0,
            'order_result': {'quantity': order_qty} if order_qty > 0 else {},
            'workflow_path': 'NORMAL'
        }

    def _node_emergency_reorder(self, state: SupplyChainState) -> dict:
        """Node 3b: Emergency path — low stock, needs immediate restocking."""
        predicted = state.get('predicted_demand', self.daily_demand)
        
        # Force higher order quantity in emergency
        emergency_qty = max(
            self.warehouse.reorder_qty,
            int(self.warehouse.max_capacity - self.warehouse.inventory)
        )
        order_qty = self.warehouse.check_reorder(predicted)
        if order_qty == 0:
            order_qty = emergency_qty

        rec = self.xai.create_record("Warehouse", "emergency_reorder", self.current_day)
        rec.workflow_path = "EMERGENCY"
        rec.action = f"EMERGENCY reorder {order_qty} units"
        rec.is_llm_decision = hasattr(self.warehouse, 'last_reasoning') and self.warehouse.last_reasoning is not None
        rec.confidence = 0.9
        rec.triggered_by = f"Inventory ({self.warehouse.inventory}) critically low"

        days_of_stock = self.warehouse.inventory / max(self.daily_demand, 1)
        rec.set_why(
            reasoning=getattr(self.warehouse, 'last_reasoning', '') or "Emergency protocol activated",
            summary=f"EMERGENCY: Only {days_of_stock:.1f} days of stock remaining. Ordering {order_qty} units to prevent stockout.",
            factors=[
                f"Inventory ({self.warehouse.inventory}) is below reorder point ({self.warehouse.reorder_point})",
                f"Only {days_of_stock:.1f} days of stock at current demand rate",
                f"Supplier is {self.supplier.status}",
                f"Logistics is {self.logistics.status}"
            ],
            alternatives=[
                {'option': 'Normal reorder (smaller quantity)',
                 'rejected_because': 'inventory too low; risk of stockout with normal order'},
                {'option': 'Wait for pending shipments',
                 'rejected_because': f'{len(self.logistics.shipments)} pending but arrival uncertain'}
            ]
        )

        self.log_event("Orchestrator", f"EMERGENCY path: ordering {order_qty} units")

        return {
            'order_placed': order_qty > 0,
            'order_result': {'quantity': order_qty},
            'workflow_path': 'EMERGENCY'
        }

    def _node_crisis_response(self, state: SupplyChainState) -> dict:
        """Node 3c: Crisis path — stockout or disruption + low inventory."""
        predicted = state.get('predicted_demand', self.daily_demand)
        crisis_qty = int(self.warehouse.max_capacity * 0.8)
        
        order_qty = self.warehouse.check_reorder(predicted)
        if order_qty == 0:
            order_qty = crisis_qty

        rec = self.xai.create_record("Warehouse", "crisis_response", self.current_day)
        rec.workflow_path = "CRISIS"
        rec.action = f"CRISIS order {order_qty} units"
        rec.is_llm_decision = hasattr(self.warehouse, 'last_reasoning') and self.warehouse.last_reasoning is not None
        rec.confidence = 0.95
        rec.triggered_by = "Stockout or near-stockout during active disruption"

        rec.set_why(
            reasoning=getattr(self.warehouse, 'last_reasoning', '') or "Crisis protocol — maximum priority restocking",
            summary=f"CRISIS MODE: Inventory at {self.warehouse.inventory}, "
                    f"supplier={'DISRUPTED' if self.supplier.status == 'disrupted' else 'active'}, "
                    f"logistics={'DISRUPTED' if self.logistics.status == 'disrupted' else 'active'}. "
                    f"Ordering maximum {order_qty} units.",
            factors=[
                f"Inventory at critical level: {self.warehouse.inventory}",
                f"Supplier: {self.supplier.status}",
                f"Logistics: {self.logistics.status}",
                f"Active disruptions: {len(self.disruption_schedule)}",
                "Customer demand cannot be met at current levels"
            ],
            alternatives=[
                {'option': 'Normal reorder',
                 'rejected_because': 'crisis conditions require maximum response'},
                {'option': 'Wait for recovery',
                 'rejected_because': 'unacceptable stockout risk; customer satisfaction at stake'}
            ]
        )

        self.log_event("Orchestrator", f"CRISIS path: maximum order {order_qty} units")

        return {
            'order_placed': order_qty > 0,
            'order_result': {'quantity': order_qty},
            'workflow_path': 'CRISIS'
        }

    def _node_supplier_decision(self, state: SupplyChainState) -> dict:
        """Node 4: Supplier processes the order."""
        if not state.get('order_placed'):
            return {'order_result': {}}

        qty = state['order_result'].get('quantity', 0)
        if qty <= 0:
            return {'order_result': {}}

        result = self.supplier.process_order(qty)

        rec = self.xai.create_record("Supplier", "order_processing", self.current_day)
        rec.workflow_path = state.get('workflow_path', '')

        if result == 0 or not result:
            rec.action = "Order REJECTED (disrupted)"
            rec.set_why(
                reasoning="Supplier is currently disrupted and cannot process orders",
                summary=f"Cannot fulfill {qty} units — supplier disrupted until recovery",
                factors=[
                    f"Supplier status: {self.supplier.status}",
                    f"Disruption schedule: {self.disruption_schedule}"
                ]
            )
            rec.confidence = 1.0
            return {'order_result': {}, 'order_placed': False}
        else:
            shipped = result.get('quantity', 0) if isinstance(result, dict) else result
            rec.action = f"Shipped {shipped} units"
            rec.is_llm_decision = hasattr(self.supplier, 'last_reasoning') and self.supplier.last_reasoning is not None
            rec.confidence = 0.85

            reasoning = getattr(self.supplier, 'last_reasoning', '') or ''
            rec.set_why(
                reasoning=reasoning or f"Processed order within capacity ({self.supplier.capacity})",
                summary=f"Fulfilled order: shipping {shipped} units with {result.get('lead_time', 2) if isinstance(result, dict) else 2}d lead time",
                factors=[
                    f"Requested: {qty} units",
                    f"Capacity: {self.supplier.capacity}",
                    f"Reliability: {self.supplier.reliability:.0%}"
                ]
            )

            return {'order_result': result if isinstance(result, dict) else {'quantity': result}}

    def _node_logistics_scheduling(self, state: SupplyChainState) -> dict:
        """Node 5: Schedule shipment."""
        order = state.get('order_result', {})
        if not order:
            return {}

        self.logistics.schedule_shipment(order)

        rec = self.xai.create_record("Logistics", "shipment_scheduling", self.current_day)
        rec.workflow_path = state.get('workflow_path', '')
        rec.is_llm_decision = hasattr(self.logistics, 'last_reasoning') and self.logistics.last_reasoning is not None
        rec.confidence = 0.8

        qty = order.get('quantity', 0)
        lt = order.get('lead_time', self.logistics.default_lead_time if hasattr(self.logistics, 'default_lead_time') else 3)
        reasoning = getattr(self.logistics, 'last_reasoning', '') or ''
        rec.action = f"Scheduled {qty} units, ETA day {self.current_day + lt}"
        rec.set_why(
            reasoning=reasoning or f"Standard scheduling with {lt}d lead time",
            summary=f"Shipment of {qty} units scheduled. Arrives day {self.current_day + lt} ({lt}d transit).",
            factors=[
                f"Quantity: {qty}",
                f"Lead time: {lt} days",
                f"Logistics status: {self.logistics.status}",
                f"Already pending: {len(self.logistics.shipments)} shipments"
            ]
        )

        return {'pending_shipments': len(self.logistics.shipments)}

    def _node_process_deliveries(self, state: SupplyChainState) -> dict:
        """Node 6: Process arriving shipments."""
        before = self.warehouse.inventory
        self.logistics.step()
        after = self.warehouse.inventory

        delivered = after - before
        if delivered > 0:
            rec = self.xai.create_record("Logistics", "delivery", self.current_day)
            rec.action = f"Delivered {delivered} units"
            rec.confidence = 1.0
            rec.workflow_path = state.get('workflow_path', '')
            rec.set_why(
                reasoning=f"Shipment arrived on schedule at day {self.current_day}",
                summary=f"Received {delivered} units. Inventory went from {before} to {after}.",
                factors=[f"Scheduled arrival matched current day"]
            )

        return {
            'inventory': self.warehouse.inventory,
            'pending_shipments': len(self.logistics.shipments)
        }

    def _node_collect_metrics(self, state: SupplyChainState) -> dict:
        """Node 7: Collect metrics and finalize step."""
        self.datacollector.collect(self)
        self._last_workflow_path = state.get('workflow_path', 'NORMAL')
        return {}

    # =========================================================================
    # Main Step — runs the LangGraph workflow
    # =========================================================================

    def step(self):
        """Execute one simulation day via LangGraph."""
        self.current_day += 1
        self._check_disruption_recovery()

        # Build initial state
        initial_state: SupplyChainState = {
            'day': self.current_day,
            'inventory': self.warehouse.inventory,
            'demand': 0,
            'predicted_demand': 0,
            'supplier_status': self.supplier.status,
            'logistics_status': self.logistics.status,
            'pending_shipments': len(self.logistics.shipments),
            'stockouts': self.stockouts,
            'order_placed': False,
            'order_result': {},
            'workflow_path': '',
            'decisions_made': []
        }

        # Run the workflow
        result = self._workflow.invoke(initial_state)
        self._last_workflow_path = result.get('workflow_path', 'NORMAL')

    # =========================================================================
    # Standard Model Methods
    # =========================================================================

    def log_event(self, agent_type, message):
        event = {'day': self.current_day, 'agent': agent_type, 'message': message}
        self.event_log.append(event)
        self.data_layer.save_simulation_log(event)

    def inject_disruption(self, agent_type, duration=3):
        end_day = self.current_day + duration
        self.disruption_schedule[agent_type] = end_day
        if agent_type == 'supplier':
            self.supplier.status = 'disrupted'
            self.log_event('Disruption', f"Supplier DISRUPTED for {duration} days")
        elif agent_type == 'logistics':
            self.logistics.status = 'disrupted'
            self.log_event('Disruption', f"Logistics DISRUPTED for {duration} days")
        return duration

    def _check_disruption_recovery(self):
        recovered = []
        for agent_type, end_day in list(self.disruption_schedule.items()):
            if self.current_day >= end_day:
                if agent_type == 'supplier' and self.supplier.status == 'disrupted':
                    self.supplier.status = 'active'
                    self.log_event('Recovery', 'Supplier recovered')
                    recovered.append(agent_type)
                elif agent_type == 'logistics' and self.logistics.status == 'disrupted':
                    self.logistics.status = 'active'
                    self.log_event('Recovery', 'Logistics recovered')
                    recovered.append(agent_type)
        for a in recovered:
            del self.disruption_schedule[a]

    def get_status(self):
        status = {
            'day': self.current_day,
            'mode': self.agent_mode,
            'workflow_path': self._last_workflow_path,
            'inventory': self.warehouse.inventory,
            'demand': self.daily_demand,
            'stockouts': self.stockouts,
            'supplier_status': self.supplier.status,
            'logistics_status': self.logistics.status,
            'pending_shipments': len(self.logistics.shipments),
        }
        try:
            llm = LLMEngine.get_instance()
            status['llm_stats'] = llm.get_stats()
        except Exception:
            pass
        try:
            status['xai_summary'] = self.xai.get_summary()
        except Exception:
            pass
        return status

    def get_agent_reasoning(self):
        reasoning = {}
        for name, agent in [('Supplier', self.supplier), ('Warehouse', self.warehouse),
                            ('Logistics', self.logistics), ('Demand', self.demand_agent)]:
            if hasattr(agent, 'last_reasoning') and agent.last_reasoning:
                reasoning[name] = agent.last_reasoning
        return reasoning

    def get_xai_decisions(self, day=None, last_n=20):
        """Get explainable decisions."""
        if day:
            return self.xai.get_decision_chain(day)
        return self.xai.get_latest_decisions(last_n)

    def explain_day(self, day):
        """Get full WHY explanation for a day."""
        return self.xai.explain_day(day)

    def get_memory_stats(self):
        stats = {}
        for name, agent in [('Supplier', self.supplier), ('Warehouse', self.warehouse),
                            ('Logistics', self.logistics), ('Demand', self.demand_agent)]:
            if hasattr(agent, 'memory'):
                stats[name] = agent.memory.get_stats()
        return stats
