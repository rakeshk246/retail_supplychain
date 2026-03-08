# agentic_agents.py
# ==============================================================================
# Phase 2+3: LLM-Powered ReAct Agents with Memory & Communication
# Uses Groq (LLaMA 3.3 70B) + ChromaDB Memory + Message Bus
# ==============================================================================

import os
import json
import random
import numpy as np
from typing import Optional

from mesa import Agent
from dotenv import load_dotenv

load_dotenv()

# LangChain imports
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

# Phase 3 imports
from memory import AgentMemory
from message_bus import MessageBus, Message


# ==============================================================================
# LLM Engine
# ==============================================================================

class GroqRateLimitError(Exception):
    """Raised when Groq API rate limit is exceeded."""
    pass


class LLMEngine:
    """Centralized LLM engine using Groq."""

    _instance = None

    def __init__(self, model=None, temperature=0.3):
        # Use env var or default to 8b-instant (500K+ tokens/day free tier vs 100K for 70b)
        self.model_name = model or os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
        self.temperature = temperature
        self.llm = None
        self.call_count = 0
        self.total_tokens = 0
        self.rate_limited = False
        self.rate_limit_message = ""
        self._initialize()

    @classmethod
    def get_instance(cls, **kwargs):
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def _initialize(self):
        if not HAS_LANGCHAIN:
            print("WARNING: langchain-groq not installed.")
            return

        api_key = os.getenv('GROQ_API_KEY')
        if not api_key or api_key == 'your-groq-key-here':
            print("WARNING: GROQ_API_KEY not set.")
            return

        try:
            self.llm = ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key
            )
            print(f"LLM Engine initialized: {self.model_name} via Groq")
        except Exception as e:
            print(f"WARNING: Failed to initialize LLM: {e}")

    @property
    def is_available(self):
        return self.llm is not None and not self.rate_limited

    def reason(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if self.rate_limited:
            raise GroqRateLimitError(self.rate_limit_message)

        if not self.llm:
            return None

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            self.call_count += 1
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.total_tokens += response.usage_metadata.get('total_tokens', 0)
            return response.content
        except Exception as e:
            error_str = str(e).lower()
            # Detect rate limit errors from Groq
            if any(kw in error_str for kw in ['rate_limit', 'rate limit', 'ratelimit',
                                                'too many requests', '429', 'quota',
                                                'tokens per', 'requests per']):
                self.rate_limited = True
                self.rate_limit_message = f"Groq API rate limit exceeded: {e}"
                print(f"\n{'='*60}")
                print(f"GROQ RATE LIMIT EXCEEDED!")
                print(f"Error: {e}")
                print(f"{'='*60}\n")
                raise GroqRateLimitError(self.rate_limit_message)
            print(f"LLM call failed: {e}")
            return None

    def get_stats(self):
        return {
            'model': self.model_name,
            'available': self.is_available,
            'rate_limited': self.rate_limited,
            'rate_limit_message': self.rate_limit_message,
            'total_calls': self.call_count,
            'total_tokens': self.total_tokens
        }



# ==============================================================================
# System Prompts (enhanced with memory/communication context)
# ==============================================================================

SUPPLIER_SYSTEM_PROMPT = """You are an intelligent Supplier Agent in a retail supply chain simulation.

Your responsibilities:
- Process purchase orders from the warehouse
- Manage production capacity and reliability
- Make decisions about order fulfillment quantity

You have access to:
1. Past decision experiences (learn from what worked before)
2. Messages from other agents (coordinate decisions)

Always respond with a JSON object containing:
{
    "reasoning": "Your step-by-step thinking",
    "action": "fulfill" or "partial_fulfill" or "reject",
    "quantity": <number of units to ship>,
    "lead_time": <estimated delivery days>,
    "confidence": <0.0-1.0>
}"""

WAREHOUSE_SYSTEM_PROMPT = """You are an intelligent Warehouse Agent in a retail supply chain simulation.

Your responsibilities:
- Manage inventory levels and reorder decisions
- Optimize safety stock based on demand patterns
- Balance between stockout risk and overstock costs

You have access to:
1. Past decision experiences (learn from what worked before)
2. Messages from other agents (e.g., demand forecasts, shipment updates)

Always respond with a JSON object containing:
{
    "reasoning": "Your step-by-step thinking",
    "action": "reorder" or "hold" or "emergency_reorder",
    "order_quantity": <number of units to order>,
    "safety_stock_adjustment": <+/- units>,
    "confidence": <0.0-1.0>
}"""

LOGISTICS_SYSTEM_PROMPT = """You are an intelligent Logistics Agent in a retail supply chain simulation.

Your responsibilities:
- Manage shipment scheduling and route optimization
- Handle disruption rerouting
- Optimize delivery lead times

You have access to:
1. Past routing decisions and outcomes
2. Alerts from other agents about inventory levels and disruptions

Always respond with a JSON object containing:
{
    "reasoning": "Your step-by-step thinking",
    "action": "schedule" or "expedite" or "reroute" or "delay",
    "lead_time_adjustment": <+/- days from default>,
    "priority": "normal" or "high" or "critical",
    "confidence": <0.0-1.0>
}"""

DEMAND_SYSTEM_PROMPT = """You are an intelligent Demand Forecasting Agent in a retail supply chain simulation.

Your responsibilities:
- Analyze demand patterns and generate accurate forecasts
- Identify anomalies and trend shifts
- Share forecasts with other agents

Always respond with a JSON object containing:
{
    "reasoning": "Your step-by-step thinking",
    "predicted_demand": <forecasted units>,
    "trend": "increasing" or "decreasing" or "stable" or "volatile",
    "anomaly_detected": true/false,
    "confidence": <0.0-1.0>
}"""


# ==============================================================================
# Helper: JSON parser
# ==============================================================================

def _parse_json(text):
    """Extract JSON from LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ==============================================================================
# Agentic Supplier Agent
# ==============================================================================

class AgenticSupplierAgent(Agent):
    """LLM-powered supplier with memory and communication."""

    def __init__(self, model, reliability=0.95, capacity=500, lead_time=2):
        super().__init__(model)
        self.status = "active"
        self.reliability = reliability
        self.capacity = capacity
        self.lead_time = lead_time
        self.total_orders = 0
        self.fulfilled_orders = 0
        self.llm_engine = LLMEngine.get_instance()
        self.memory = AgentMemory("Supplier", persist_dir='./chroma_db')
        self.bus = MessageBus.get_instance()
        self.last_reasoning = None

    def process_order(self, quantity):
        if self.status == "disrupted":
            self.model.log_event("Supplier", "Unable to fulfill — DISRUPTED")
            # Broadcast disruption alert
            self.bus.broadcast_alert("Supplier", "disruption_alert", {
                'agent': 'supplier', 'status': 'disrupted',
                'day': self.model.current_day
            })
            return 0

        llm_decision = self._llm_decide(quantity)
        if llm_decision:
            result = self._execute_llm_decision(llm_decision, quantity)
        else:
            result = self._rule_based_order(quantity)

        # Store episode in memory
        outcome = {'success': result != 0, 'quantity_fulfilled': result.get('quantity', 0) if isinstance(result, dict) else 0}
        self.memory.store_episode(
            situation=f"Order for {quantity} units, capacity={self.capacity}, status={self.status}",
            decision=f"Fulfilled {outcome['quantity_fulfilled']} units",
            outcome=outcome,
            day=self.model.current_day,
            metadata={'decision_type': 'order_fulfillment'}
        )

        # Send order confirmation
        if isinstance(result, dict) and result.get('quantity', 0) > 0:
            self.bus.send_direct("Supplier", "Logistics", "order_confirmation", {
                'quantity': result['quantity'],
                'lead_time': result.get('lead_time', self.lead_time),
                'day': self.model.current_day
            })

        return result

    def _llm_decide(self, quantity):
        # Recall similar past decisions
        situation = f"Order for {quantity} units, inventory={self.model.warehouse.inventory}"
        past_episodes = self.memory.recall_similar(situation, n_results=3)
        memory_context = self.memory.format_for_prompt(past_episodes)

        # Get inter-agent messages
        bus_context = self.bus.format_for_prompt("Supplier", limit=5)

        context = (
            f"Order received: {quantity} units\n"
            f"Status: {self.status} | Capacity: {self.capacity} | Reliability: {self.reliability:.0%}\n"
            f"Lead time: {self.lead_time}d | Day: {self.model.current_day}\n"
            f"Warehouse inventory: {self.model.warehouse.inventory}\n"
            f"Pending shipments: {len(self.model.logistics.shipments)}\n\n"
            f"{memory_context}\n\n"
            f"{bus_context}"
        )

        response = self.llm_engine.reason(SUPPLIER_SYSTEM_PROMPT, context)
        if response:
            decision = _parse_json(response)
            self.last_reasoning = decision.get('reasoning', '')
            return decision
        return None

    def _execute_llm_decision(self, decision, requested_qty):
        qty = min(decision.get('quantity', requested_qty), self.capacity)
        lt = decision.get('lead_time', self.lead_time)
        self.total_orders += 1
        self.fulfilled_orders += 1
        self.model.log_event("Supplier [AI]",
            f"{decision.get('action', 'fulfill')}: {qty} units (lt:{lt}d) | {decision.get('reasoning', '')[:60]}")
        return {'quantity': qty, 'lead_time': lt}

    def _rule_based_order(self, quantity):
        actual_qty = min(quantity, self.capacity)
        if random.random() > self.reliability:
            actual_qty = int(actual_qty * random.uniform(0.7, 0.95))
        self.model.log_event("Supplier", f"Processed order: {actual_qty} units")
        self.total_orders += 1
        self.fulfilled_orders += 1
        return {'quantity': actual_qty, 'lead_time': self.lead_time}

    def step(self):
        pass


# ==============================================================================
# Agentic Warehouse Agent
# ==============================================================================

class AgenticWarehouseAgent(Agent):
    """LLM-powered warehouse with memory and communication."""

    def __init__(self, model, initial_inventory=500,
                 reorder_point=200, reorder_qty=300, max_capacity=1000):
        super().__init__(model)
        self.inventory = initial_inventory
        self.reorder_point = reorder_point
        self.reorder_qty = reorder_qty
        self.max_capacity = max_capacity
        self.target_stock = 500
        self.llm_engine = LLMEngine.get_instance()
        self.memory = AgentMemory("Warehouse", persist_dir='./chroma_db')
        self.bus = MessageBus.get_instance()
        self.last_reasoning = None
        self.demand_history = []

    def fulfill_demand(self, demand):
        self.demand_history.append(demand)
        fulfilled = min(self.inventory, demand)
        self.inventory -= fulfilled
        if fulfilled < demand:
            self.model.stockouts += 1
            self.model.log_event("Warehouse",
                f"STOCKOUT: {fulfilled}/{demand} (short {demand - fulfilled})")
            # Alert other agents
            self.bus.broadcast_alert("Warehouse", "inventory_alert", {
                'type': 'stockout', 'inventory': self.inventory,
                'shortfall': demand - fulfilled, 'day': self.model.current_day
            })
        else:
            self.model.log_event("Warehouse",
                f"Fulfilled {fulfilled} | Remaining: {self.inventory}")
            # Low stock warning
            if self.inventory < self.reorder_point:
                self.bus.broadcast_alert("Warehouse", "inventory_alert", {
                    'type': 'low_stock', 'inventory': self.inventory,
                    'reorder_point': self.reorder_point, 'day': self.model.current_day
                })
        return fulfilled

    def check_reorder(self, predicted_demand):
        llm_decision = self._llm_decide(predicted_demand)
        if llm_decision:
            order_qty = self._execute_llm_decision(llm_decision)
        else:
            order_qty = self._rule_based_reorder(predicted_demand)

        # Store episode
        self.memory.store_episode(
            situation=f"Inventory={self.inventory}, predicted={predicted_demand:.0f}, "
                      f"reorder_pt={self.reorder_point}",
            decision=f"{'Reorder ' + str(order_qty) if order_qty > 0 else 'Hold'}",
            outcome={'success': True, 'order_qty': order_qty,
                     'inventory_before': self.inventory},
            day=self.model.current_day,
            metadata={'decision_type': 'reorder'}
        )

        # Send order request to supplier
        if order_qty > 0:
            self.bus.send_direct("Warehouse", "Supplier", "order_request", {
                'quantity': order_qty, 'urgency': 'high' if self.inventory < 100 else 'normal',
                'day': self.model.current_day
            })

        return order_qty

    def _llm_decide(self, predicted_demand):
        recent_demands = self.demand_history[-14:]
        avg_demand = np.mean(recent_demands) if recent_demands else 100

        situation = f"Inventory={self.inventory}, predicted_demand={predicted_demand:.0f}"
        past_episodes = self.memory.recall_similar(situation, n_results=3)
        memory_context = self.memory.format_for_prompt(past_episodes)
        bus_context = self.bus.format_for_prompt("Warehouse", limit=5)

        context = (
            f"Inventory: {self.inventory} | Reorder point: {self.reorder_point}\n"
            f"Max capacity: {self.max_capacity} | Default reorder qty: {self.reorder_qty}\n"
            f"Predicted demand: {predicted_demand:.0f} | Avg demand (14d): {avg_demand:.0f}\n"
            f"Recent demands: {[int(d) for d in recent_demands[-7:]]}\n"
            f"Days of stock: {self.inventory / max(avg_demand, 1):.1f}\n"
            f"Supplier: {self.model.supplier.status} | Logistics: {self.model.logistics.status}\n"
            f"Pending shipments: {len(self.model.logistics.shipments)}\n"
            f"Day: {self.model.current_day}\n\n"
            f"{memory_context}\n\n"
            f"{bus_context}"
        )

        response = self.llm_engine.reason(WAREHOUSE_SYSTEM_PROMPT, context)
        if response:
            decision = _parse_json(response)
            self.last_reasoning = decision.get('reasoning', '')
            return decision
        return None

    def _execute_llm_decision(self, decision):
        action = decision.get('action', 'hold')
        order_qty = decision.get('order_quantity', 0)
        if action == 'hold':
            self.model.log_event("Warehouse [AI]",
                f"Hold | {decision.get('reasoning', '')[:60]}")
            return 0
        order_qty = min(max(order_qty, 0), self.max_capacity - self.inventory)
        priority = "EMERGENCY" if action == 'emergency_reorder' else "Normal"
        self.model.log_event("Warehouse [AI]",
            f"{priority} reorder: {order_qty} | {decision.get('reasoning', '')[:60]}")
        return order_qty

    def _rule_based_reorder(self, predicted_demand):
        """Smart fallback — uses demand trends, memory, and bus context.
        
        Unlike the basic WarehouseAgent which uses a fixed threshold,
        this adapts based on:
        1. Demand trend (rising/falling/volatile)
        2. Supplier/logistics status
        3. Past memory success rates
        4. Number of pending shipments
        """
        recent = self.demand_history[-14:]
        avg_demand = sum(recent) / len(recent) if recent else predicted_demand
        
        # Calculate demand trend
        if len(recent) >= 7:
            recent_avg = sum(recent[-7:]) / 7
            older_avg = sum(recent[:7]) / max(len(recent[:7]), 1)
            trend_ratio = recent_avg / max(older_avg, 1)
        else:
            trend_ratio = 1.0

        # Calculate days of stock remaining
        days_of_stock = self.inventory / max(avg_demand, 1)
        
        # Check disruption status from bus messages
        supplier_disrupted = self.model.supplier.status == 'disrupted'
        logistics_disrupted = self.model.logistics.status == 'disrupted'
        pending_shipments = len(self.model.logistics.shipments)
        
        # Adaptive reorder point based on conditions
        adaptive_reorder = self.reorder_point
        if trend_ratio > 1.15:
            # Demand is rising — raise the reorder point
            adaptive_reorder = int(self.reorder_point * 1.3)
        if supplier_disrupted or logistics_disrupted:
            # Active disruption — be more cautious
            adaptive_reorder = int(self.reorder_point * 1.5)
        
        # Decision logic: much smarter than basic "if inv < reorder_point"
        if self.inventory < adaptive_reorder:
            # Calculate smart order quantity
            safety_days = 5 if supplier_disrupted else 3
            target = avg_demand * safety_days + predicted_demand
            
            # Adjust for trend
            if trend_ratio > 1.1:
                target *= 1.2  # Order more if demand rising
            
            # Subtract pending deliveries to avoid over-ordering
            pending_qty = sum(
                s.get('quantity', 0) if isinstance(s, dict) else getattr(s, 'quantity', 0)
                for s in self.model.logistics.shipments
            )
            target = max(0, target - pending_qty)
            
            order_qty = max(self.reorder_qty, int(target))
            order_qty = min(order_qty, self.max_capacity - self.inventory)
            
            # Log with context
            reason = []
            if supplier_disrupted:
                reason.append("supplier disrupted")
            if trend_ratio > 1.1:
                reason.append(f"demand rising {trend_ratio:.0%}")
            if days_of_stock < 2:
                reason.append(f"only {days_of_stock:.1f} days stock")
            reason_str = ", ".join(reason) if reason else "below reorder point"
            
            self.model.log_event("Warehouse",
                f"Smart reorder: {order_qty} units ({reason_str})")
            return order_qty
        
        # Proactive ordering: even above reorder point, order if conditions warrant
        if days_of_stock < 3 and trend_ratio > 1.1 and pending_shipments == 0:
            order_qty = int(avg_demand * 3)
            order_qty = min(order_qty, self.max_capacity - self.inventory)
            if order_qty > 0:
                self.model.log_event("Warehouse",
                    f"Proactive reorder: {order_qty} (trend rising, low buffer)")
                return order_qty
        
        return 0

    def receive_shipment(self, quantity):
        space = self.max_capacity - self.inventory
        accepted = min(quantity, space)
        self.inventory += accepted
        self.model.log_event("Warehouse", f"Received {accepted} | Stock: {self.inventory}")

    def step(self):
        pass


# ==============================================================================
# Agentic Logistics Agent
# ==============================================================================

class AgenticLogisticsAgent(Agent):
    """LLM-powered logistics with memory and communication."""

    def __init__(self, model, default_lead_time=3):
        super().__init__(model)
        self.status = "active"
        self.default_lead_time = default_lead_time
        self.shipments = []
        self.total_shipments = 0
        self.delivered_shipments = 0
        self.llm_engine = LLMEngine.get_instance()
        self.memory = AgentMemory("Logistics", persist_dir='./chroma_db')
        self.bus = MessageBus.get_instance()
        self.last_reasoning = None

    def schedule_shipment(self, quantity, lead_time=None):
        if isinstance(quantity, dict):
            lead_time = quantity.get('lead_time', self.default_lead_time)
            quantity = quantity.get('quantity', 0)
        if lead_time is None:
            lead_time = self.default_lead_time

        # Try LLM decision
        llm_decision = self._llm_decide(quantity, lead_time)
        if llm_decision:
            lead_time = self._apply_llm_decision(llm_decision, lead_time)
        elif self.status == "disrupted":
            lead_time += 2
            self.model.log_event("Logistics", "Disruption delay: +2 days")

        arrival_day = self.model.current_day + lead_time
        self.shipments.append({
            'quantity': quantity, 'arrival_day': arrival_day,
            'scheduled_day': self.model.current_day
        })
        self.total_shipments += 1

        # Store episode
        self.memory.store_episode(
            situation=f"Shipment {quantity} units, base_lt={lead_time}d, status={self.status}",
            decision=f"Scheduled arrival day {arrival_day}",
            outcome={'lead_time': lead_time, 'disrupted': self.status == 'disrupted'},
            day=self.model.current_day,
            metadata={'decision_type': 'scheduling'}
        )

        # Notify warehouse
        self.bus.send_direct("Logistics", "Warehouse", "shipment_update", {
            'quantity': quantity, 'arrival_day': arrival_day,
            'status': 'in_transit', 'day': self.model.current_day
        })

        self.model.log_event("Logistics",
            f"Shipment {quantity} units → day {arrival_day} (lt:{lead_time}d)")

    def _llm_decide(self, quantity, base_lead_time):
        situation = f"Shipment {quantity} units, inventory={self.model.warehouse.inventory}"
        past = self.memory.recall_similar(situation, n_results=3)
        memory_context = self.memory.format_for_prompt(past)
        bus_context = self.bus.format_for_prompt("Logistics", limit=5)

        context = (
            f"Shipment: {quantity} units | Base lead time: {base_lead_time}d\n"
            f"Status: {self.status} | Pending: {len(self.shipments)}\n"
            f"Warehouse inventory: {self.model.warehouse.inventory}\n"
            f"Demand: {self.model.daily_demand} | Day: {self.model.current_day}\n"
            f"Days of stock: {self.model.warehouse.inventory / max(self.model.daily_demand, 1):.1f}\n\n"
            f"{memory_context}\n\n{bus_context}"
        )

        response = self.llm_engine.reason(LOGISTICS_SYSTEM_PROMPT, context)
        if response:
            decision = _parse_json(response)
            self.last_reasoning = decision.get('reasoning', '')
            return decision
        return None

    def _apply_llm_decision(self, decision, base_lead_time):
        action = decision.get('action', 'schedule')
        adjustment = decision.get('lead_time_adjustment', 0)

        if action == 'expedite':
            lt = max(1, base_lead_time - abs(adjustment))
            self.model.log_event("Logistics [AI]", f"EXPEDITED — lt:{lt}d")
            return lt
        elif action == 'reroute' and self.status == 'disrupted':
            lt = base_lead_time + 1
            self.model.log_event("Logistics [AI]", f"Rerouted — lt:{lt}d")
            return lt
        elif action == 'delay':
            lt = base_lead_time + abs(adjustment)
            self.model.log_event("Logistics [AI]", f"Delayed — lt:{lt}d")
            return lt

        if self.status == "disrupted":
            return base_lead_time + 2
        return base_lead_time

    def step(self):
        arrived = [s for s in self.shipments if s['arrival_day'] <= self.model.current_day]
        pending = [s for s in self.shipments if s['arrival_day'] > self.model.current_day]
        self.shipments = pending

        for shipment in arrived:
            self.model.warehouse.receive_shipment(shipment['quantity'])
            self.delivered_shipments += 1
            transit = shipment['arrival_day'] - shipment['scheduled_day']
            self.model.log_event("Logistics",
                f"Delivered {shipment['quantity']} units (transit: {transit}d)")
            # Notify delivery
            self.bus.send_direct("Logistics", "Warehouse", "shipment_update", {
                'quantity': shipment['quantity'], 'status': 'delivered',
                'day': self.model.current_day
            })

        if pending:
            self.model.log_event("Logistics", f"{len(pending)} in transit")


# ==============================================================================
# Agentic Demand Agent
# ==============================================================================

class AgenticDemandAgent(Agent):
    """LLM-powered demand forecasting with memory and communication."""

    def __init__(self, model, forecaster):
        super().__init__(model)
        self.forecaster = forecaster
        self.llm_engine = LLMEngine.get_instance()
        self.memory = AgentMemory("Demand", persist_dir='./chroma_db')
        self.bus = MessageBus.get_instance()
        self.last_reasoning = None
        self.recent_demands = []

    def step(self):
        try:
            base_prediction = self.forecaster.predict_next()
        except Exception:
            base_prediction = 100.0

        llm_decision = self._llm_decide(base_prediction)
        if llm_decision:
            demand = self._apply_llm_forecast(llm_decision, base_prediction)
        else:
            variance = base_prediction * 0.2
            demand = max(1, int(base_prediction + np.random.normal(0, variance)))

        self.model.daily_demand = demand
        self.recent_demands.append(demand)

        # Store episode
        self.memory.store_episode(
            situation=f"Base prediction={base_prediction:.0f}, recent_avg={np.mean(self.recent_demands[-7:]) if self.recent_demands else 0:.0f}",
            decision=f"Forecast: {demand}",
            outcome={'demand': demand, 'base_prediction': base_prediction},
            day=self.model.current_day,
            metadata={'decision_type': 'forecast'}
        )

        # Share forecast with other agents
        self.bus.broadcast_alert("Demand", "demand_forecast", {
            'predicted_demand': demand,
            'trend': llm_decision.get('trend', 'stable') if llm_decision else 'stable',
            'day': self.model.current_day
        })

        self.model.log_event("Demand", f"Daily demand: {demand} units")

    def _llm_decide(self, base_prediction):
        recent = self.recent_demands[-14:]
        avg = np.mean(recent) if recent else base_prediction

        situation = f"Prediction={base_prediction:.0f}, avg={avg:.0f}"
        past = self.memory.recall_similar(situation, n_results=3)
        memory_context = self.memory.format_for_prompt(past)
        bus_context = self.bus.format_for_prompt("Demand", limit=3)

        context = (
            f"Base model prediction: {base_prediction:.0f}\n"
            f"Recent demands: {[int(d) for d in recent[-7:]]}\n"
            f"14-day average: {avg:.0f} | Day: {self.model.current_day}\n"
            f"Inventory: {self.model.warehouse.inventory}\n\n"
            f"{memory_context}\n\n{bus_context}"
        )

        response = self.llm_engine.reason(DEMAND_SYSTEM_PROMPT, context)
        if response:
            decision = _parse_json(response)
            self.last_reasoning = decision.get('reasoning', '')
            return decision
        return None

    def _apply_llm_forecast(self, decision, base_prediction):
        predicted = decision.get('predicted_demand', base_prediction)
        predicted = max(1, min(int(predicted), int(base_prediction * 3)))
        trend = decision.get('trend', 'stable')
        anomaly = decision.get('anomaly_detected', False)
        if anomaly:
            self.model.log_event("Demand [AI]",
                f"ANOMALY — forecast: {predicted} (trend: {trend})")
        else:
            self.model.log_event("Demand [AI]",
                f"forecast: {predicted} (trend: {trend})")
        return predicted
