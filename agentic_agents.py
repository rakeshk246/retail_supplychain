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

# Phase 5: Real-time intelligence (weather + news + LLM analysis)
try:
    from news_search import get_intelligence_analyzer
    HAS_INTEL = True
except ImportError:
    HAS_INTEL = False


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
    """Extract JSON from LLM response, handling markdown code fences."""
    if not text:
        return {}
    import re
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: extract first {...} block (handles prose + JSON mixed responses)
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
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

        # Proactively broadcast current capacity status so all agents plan accordingly
        if isinstance(result, dict) and result.get('quantity', 0) > 0:
            qty = result['quantity']
            lt = result.get('lead_time', self.lead_time)
            remaining_capacity = self.capacity - qty
            utilization_pct = (qty / self.capacity) * 100
            self.bus.send_direct("Supplier", "Warehouse", "order_confirmation", {
                'order_qty': qty,
                'status': 'processing',
                'available_capacity': remaining_capacity,
                'reasoning': self.last_reasoning if self.last_reasoning else "Sufficient capacity to fulfill request.",
                'action_requested': f"Acknowledge order and prepare receiving docks.",
                'day': self.model.current_day
            })
            self.bus.send_direct("Supplier", "Logistics", "transport_request", {
                'order_qty': qty,
                'pickup_ready_in_days': lt,
                'risk_factor': f"{1.0 - self.reliability:.0%}",
                'reasoning': self.last_reasoning if self.last_reasoning else "Order processed. Requires transport.",
                'action_requested': f"Schedule urgent transport slot before 18:00 on Day {self.model.current_day + lt}.",
                'day': self.model.current_day
            })
            # I3: Proactive capacity alert — all agents now know supplier headroom
            self.bus.broadcast_alert("Supplier", "capacity_status", {
                'capacity_total': self.capacity,
                'capacity_used_this_order': qty,
                'capacity_remaining': remaining_capacity,
                'utilization_pct': round(utilization_pct, 1),
                'reliability': self.reliability,
                'status': 'WARNING: Near capacity' if utilization_pct > 80 else 'Normal',
                'day': self.model.current_day
            })
            if utilization_pct > 80:
                self.model.log_event("Supplier",
                    f"⚠️ Capacity warning: {utilization_pct:.0f}% utilized — remaining {remaining_capacity} units")

        return result

    def _llm_decide(self, quantity):
        # Recall similar past decisions
        situation = f"Order for {quantity} units, inventory={self.model.warehouse.inventory}"
        past_episodes = self.memory.recall_similar(situation, n_results=3)
        memory_context = self.memory.format_for_prompt(past_episodes)

        # Get inter-agent messages
        bus_context = self.bus.format_for_prompt("Supplier", limit=5)

        # NEW: Get intelligence context
        intel_context = ""
        if hasattr(self.model, 'demand_agent') and hasattr(self.model.demand_agent, 'last_analysis'):
            analysis = self.model.demand_agent.last_analysis
            if analysis:
                intel_context = (
                    f"SUPPLY CHAIN RISK:\n"
                    f"Risk Level: {analysis.get('risk_level', 'low').upper()}\n"
                    f"Weather Impact: {analysis.get('weather_impact', 'None')}\n"
                    f"News: {analysis.get('news_impact', 'None')}\n"
                    f"Alert: {analysis.get('alert_message', 'None')}\n\n"
                    f"Given this risk level, should we:\n"
                    f"- Prioritize fulfilling orders (maintain service)?\n"
                    f"- Reduce fulfillment to preserve capacity?\n"
                    f"- Warn warehouse about potential delays?\n"
                )

        context = (
            f"Order received: {quantity} units\n"
            f"Status: {self.status} | Capacity: {self.capacity} | Reliability: {self.reliability:.0%}\n"
            f"Lead time: {self.lead_time}d | Day: {self.model.current_day}\n"
            f"Warehouse inventory: {self.model.warehouse.inventory}\n"
            f"Pending shipments: {len(self.model.logistics.shipments)}\n\n"
            f"{memory_context}\n\n"
            f"{bus_context}\n\n"
            f"{intel_context}"
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
            shortfall = demand - fulfilled
            self.bus.broadcast_alert("Warehouse", "inventory_alert", {
                'type': 'stockout', 
                'current_inventory': self.inventory,
                'missed_demand': shortfall,
                'financial_impact': f"-${shortfall * 45}",
                'reasoning': "Demand exceeded available stock severely.",
                'action_requested': "Emergency fulfillment required from supplier. Halt promotions.",
                'day': self.model.current_day
            })
        else:
            self.model.log_event("Warehouse",
                f"Fulfilled {fulfilled} | Remaining: {self.inventory}")
            if self.inventory < self.reorder_point:
                self.bus.broadcast_alert("Warehouse", "inventory_alert", {
                    'type': 'low_stock', 
                    'current_inventory': self.inventory,
                    'reorder_threshold': self.reorder_point,
                    'buffer_status': f"{(self.inventory / self.reorder_point) * 100:.0f}%",
                    'reasoning': "Inventory fell below critical threshold.",
                    'action_requested': "Supplier prepare for incoming high-volume order.",
                    'day': self.model.current_day
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
            recent_avg = np.mean(self.demand_history[-7:]) if len(self.demand_history) >= 7 else predicted_demand
            self.bus.send_direct("Warehouse", "Supplier", "order_request", {
                'order_qty': order_qty,
                'current_inventory': self.inventory,
                'projected_shortfall': max(0, int((recent_avg * 3) - self.inventory)),
                'warehouse_utilization': f"{(self.inventory / self.max_capacity) * 100:.0f}%",
                'reasoning': self.last_reasoning if self.last_reasoning else "Calculated reorder needed based on trends.",
                'action_requested': f"Confirm and fulfill {order_qty} units within standard lead time.",
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

        # NEW: Get intelligence context
        intel_context = ""
        if hasattr(self.model, 'demand_agent') and hasattr(self.model.demand_agent, 'last_analysis'):
            analysis = self.model.demand_agent.last_analysis
            if analysis:
                intel_context = (
                    f"EXTERNAL INTELLIGENCE:\n"
                    f"Risk Level: {analysis.get('risk_level', 'low').upper()}\n"
                    f"Risk Score: {analysis.get('risk_score', 0):.0%}\n"
                    f"Weather Impact: {analysis.get('weather_impact', 'None')}\n"
                    f"News Impact: {analysis.get('news_impact', 'None')}\n"
                    f"Demand Adjustment: ×{analysis.get('demand_adjustment', 1.0):.1f}\n"
                    f"Logistics Delay: +{analysis.get('logistics_delay_days', 0)} days\n"
                    f"Supplier Reliability: {analysis.get('reliability_adjustment', 0):.0%}\n"
                    f"Recommendation: {analysis.get('recommendation', 'Normal operations')}\n"
                )

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
            f"{bus_context}\n\n"
            f"{intel_context}\n"
            f"Given the intelligence data above, how much should we reorder?\n"
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
        delay_probability = "15% due to weather" if self.status == "disrupted" else "2% nominal"
        self.bus.send_direct("Logistics", "Warehouse", "shipment_update", {
            'qty': quantity, 
            'eta_days': lead_time,
            'status': 'in transit', 
            'delay_risk': delay_probability,
            'reasoning': self.last_reasoning if self.last_reasoning else "Standard routing applied.",
            'action_requested': f"Ensure receiving dock is available on Day {arrival_day}.",
            'day': self.model.current_day
        })

        self.model.log_event("Logistics",
            f"Shipment {quantity} units → day {arrival_day} (lt:{lead_time}d)")

    def _llm_decide(self, quantity, base_lead_time):
        situation = f"Shipment {quantity} units, inventory={self.model.warehouse.inventory}"
        past = self.memory.recall_similar(situation, n_results=3)
        memory_context = self.memory.format_for_prompt(past)
        bus_context = self.bus.format_for_prompt("Logistics", limit=5)

        # NEW: Get intelligence context
        intel_context = ""
        if hasattr(self.model, 'demand_agent') and hasattr(self.model.demand_agent, 'last_analysis'):
            analysis = self.model.demand_agent.last_analysis
            if analysis:
                weather_raw = analysis.get('weather_raw', {})
                intel_context = (
                    f"LOGISTICS WEATHER IMPACT:\n"
                    f"Current: {weather_raw.get('description', 'Unknown')}\n"
                    f"Wind: {weather_raw.get('wind_speed', 0)} km/h\n"
                    f"Rain: {weather_raw.get('rain', 0)}mm\n"
                    f"Forecast Delay: +{analysis.get('logistics_delay_days', 0)} days\n\n"
                    f"Decision options:\n"
                    f"1. Expedite (+cost, -delay)\n"
                    f"2. Normal route (+risk of forecast delay)\n"
                    f"3. Delay (reduce shipping cost, warehouse can wait)\n"
                )

        context = (
            f"Shipment: {quantity} units | Base lead time: {base_lead_time}d\n"
            f"Status: {self.status} | Pending: {len(self.shipments)}\n"
            f"Warehouse inventory: {self.model.warehouse.inventory}\n"
            f"Demand: {getattr(self.model, 'daily_demand', getattr(self.model.warehouse, 'demand_history', [100])[-1] if hasattr(self.model.warehouse, 'demand_history') and self.model.warehouse.demand_history else 100)} | Day: {self.model.current_day}\n"
            f"Days of stock: {self.model.warehouse.inventory / max(getattr(self.model, 'daily_demand', 100), 1):.1f}\n\n"
            f"{memory_context}\n\n"
            f"{bus_context}\n\n"
            f"{intel_context}"
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
        # Phase 5: Intelligence analyzer (weather + news + LLM)
        self.intel = get_intelligence_analyzer() if HAS_INTEL else None
        self.last_analysis = None

    def step(self):
        # Phase 5: Gather weather + news intelligence (every 5 days or day 1)
        if self.intel and (self.model.current_day % 5 == 1 or self.model.current_day == 1):
            try:
                analysis = self.intel.analyze_with_llm(
                    self.llm_engine, self.model.current_day
                )
                self.last_analysis = analysis

                # Apply supply chain impacts
                reliability_adj = analysis.get('reliability_adjustment', 0)
                if reliability_adj < 0:
                    original_rel = self.model.supplier.reliability
                    self.model.supplier.reliability = max(
                        0.5, original_rel + reliability_adj
                    )
                    self.model.log_event("Intelligence",
                        f"📡 Risk: {analysis.get('risk_level', '?').upper()} — "
                        f"Supplier reliability: {original_rel:.0%} → {self.model.supplier.reliability:.0%}")

                # Broadcast intelligence to all agents
                weather_raw = analysis.get('weather_raw', {})
                self.bus.broadcast_alert("System", "weather_update", {
                    'emoji': weather_raw.get('emoji', '🌡️') if weather_raw else '🌡️',
                    'description': weather_raw.get('description', 'N/A') if weather_raw else 'N/A',
                    'temperature': weather_raw.get('temperature', 0) if weather_raw else 0,
                    'severity': weather_raw.get('severity', 'normal') if weather_raw else 'normal',
                    'impact': analysis.get('weather_impact', 'No data'),
                    'day': self.model.current_day
                })

                if analysis.get('risk_level', 'low') != 'low':
                    self.bus.broadcast_alert("Intelligence", "risk_alert", {
                        'risk_level': analysis.get('risk_level', 'low'),
                        'risk_score': analysis.get('risk_score', 0),
                        'recommendation': analysis.get('recommendation', ''),
                        'alert_message': analysis.get('alert_message', ''),
                        'day': self.model.current_day
                    })

            except Exception as e:
                print(f"Intelligence step failed: {e}")

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

        # I2: Enforce intelligence demand_adjustment numerically (not just as text in prompt)
        if self.last_analysis:
            adj = self.last_analysis.get('demand_adjustment', 1.0)
            if adj and abs(adj - 1.0) > 0.05:  # Only act on meaningful adjustments (>5%)
                adjusted = int(demand * adj)
                self.model.log_event("Intelligence",
                    f"📊 Demand adjusted ×{adj:.2f} ({demand} → {adjusted}) "
                    f"based on {self.last_analysis.get('risk_level','?')} risk level")
                demand = max(1, adjusted)

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
        confidence = llm_decision.get('confidence', 0.85) if llm_decision else 0.85
        last_fc = getattr(self, 'last_forecast', demand)
        pct_change = ((demand - last_fc) / max(last_fc, 1)) * 100
        
        self.bus.broadcast_alert("Demand", "demand_forecast", {
            'forecast': demand,
            'change_vs_last': f"{pct_change:+.1f}%",
            'trend': llm_decision.get('trend', 'stable') if llm_decision else 'stable',
            'confidence': f"{confidence:.0%}",
            'reasoning': self.last_reasoning if self.last_reasoning else "Computed by baseline moving average.",
            'action_requested': "Warehouse: review safety stock. Logistics: ensure capacity.",
            'day': self.model.current_day
        })
        self.last_forecast = demand

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

        # Add external intelligence context (weather + news)
        if self.intel and self.last_analysis:
            intel_ctx = self.intel.get_context_for_agent_prompt()
            context += f"\n\n{intel_ctx}"

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
