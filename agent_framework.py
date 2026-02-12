from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

import os
import numpy as np

class LLMReasoningMixin:
    """Mixin for LLM-based agent reasoning"""
    
    def __init__(self):
        # Initialize LangChain with Claude
        # Note: Set ANTHROPIC_API_KEY environment variable
        try:
            self.llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.3
            )
        except:
            self.llm = None  # Fallback to rule-based
    
    def llm_decision(self, context, question):
        """Use LLM for intelligent decision making"""
        if self.llm is None:
            return None
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an AI agent in a supply chain system.
            
Context: {context}

Question: {question}

Provide a brief, actionable decision (1-2 sentences).
"""
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(context=context, question=question)
            )
            return response.content
        except:
            return None


class SupplierAgent(Agent, LLMReasoningMixin):
    """Supplier agent with LLM reasoning"""
    
    def __init__(self, unique_id, model):
        Agent.__init__(self, unique_id, model)
        LLMReasoningMixin.__init__(self)
        
        self.reliability = 0.95
        self.capacity = 500
        self.lead_time = 2
        self.status = 'active'
        self.pending_orders = []
    
    def process_order(self, quantity):
        """Process incoming order with intelligent reasoning"""
        if self.status == 'disrupted':
            return None
        
        # Rule-based decision
        if quantity > self.capacity:
            actual_qty = self.capacity
            self.model.log_event('supplier', f"Partial fulfillment: {actual_qty}/{quantity}")
        else:
            actual_qty = quantity
            self.model.log_event('supplier', f"Order accepted: {actual_qty}")
        
        # LLM reasoning for edge cases
        if quantity > self.capacity * 1.5:
            context = f"Order: {quantity}, Capacity: {self.capacity}, Status: {self.status}"
            decision = self.llm_decision(
                context, 
                "Should we recommend finding alternative suppliers?"
            )
            if decision:
                self.model.log_event('supplier_llm', decision)
        
        return {'quantity': actual_qty, 'lead_time': self.lead_time}
    
    def step(self):
        """Agent step function"""
        pass


class WarehouseAgent(Agent, LLMReasoningMixin):
    """Warehouse agent managing inventory"""
    
    def __init__(self, unique_id, model):
        Agent.__init__(self, unique_id, model)
        LLMReasoningMixin.__init__(self)
        
        self.inventory = 200
        self.max_capacity = 1000
        self.reorder_point = 100
        self.target_stock = 300
    
    def check_reorder(self, predicted_demand):
        """Intelligent reorder decision"""
        if self.inventory < self.reorder_point:
            buffer = predicted_demand * 1.5  # Safety stock
            order_qty = min(
                self.target_stock - self.inventory + buffer,
                self.max_capacity - self.inventory
            )
            
            # LLM reasoning for complex scenarios
            if self.inventory < 50:
                context = f"Critical stock: {self.inventory}, Predicted demand: {predicted_demand}"
                decision = self.llm_decision(
                    context,
                    "Should we expedite this order or find alternative sources?"
                )
                if decision:
                    self.model.log_event('warehouse_llm', decision)
            
            return order_qty
        return 0
    
    def fulfill_demand(self, demand):
        """Fulfill customer demand"""
        fulfilled = min(self.inventory, demand)
        self.inventory -= fulfilled
        
        if fulfilled < demand:
            self.model.log_event('warehouse', f"Stock-out: {fulfilled}/{demand} fulfilled")
            self.model.stockouts += 1
        
        return fulfilled
    
    def receive_shipment(self, quantity):
        """Receive incoming shipment"""
        self.inventory = min(self.inventory + quantity, self.max_capacity)
        self.model.log_event('warehouse', f"Received: {quantity}, Stock: {self.inventory}")
    
    def step(self):
        """Agent step function"""
        pass


class LogisticsAgent(Agent, LLMReasoningMixin):
    """Logistics agent handling transportation"""
    
    def __init__(self, unique_id, model):
        Agent.__init__(self, unique_id, model)
        LLMReasoningMixin.__init__(self)
        
        self.status = 'active'
        self.shipments_in_transit = []
    
    def schedule_shipment(self, order):
        """Schedule shipment with routing optimization"""
        if self.status == 'disrupted':
            order['lead_time'] += 2
            self.model.log_event('logistics', "Shipment delayed due to disruption")
        
        shipment = {
            'quantity': order['quantity'],
            'arrival_day': self.model.current_day + order['lead_time']
        }
        
        self.shipments_in_transit.append(shipment)
        self.model.log_event('logistics', f"Scheduled: {order['quantity']} units")
        
        return shipment
    
    def update_shipments(self):
        """Update shipments and check arrivals"""
        arrived = []
        remaining = []
        
        for shipment in self.shipments_in_transit:
            if shipment['arrival_day'] <= self.model.current_day:
                arrived.append(shipment)
                self.model.log_event('logistics', f"Arrived: {shipment['quantity']} units")
            else:
                remaining.append(shipment)
        
        self.shipments_in_transit = remaining
        return arrived
    
    def step(self):
        """Agent step function"""
        arrived = self.update_shipments()
        
        # Deliver to warehouse
        if arrived:
            warehouse = self.model.get_agent_by_type(WarehouseAgent)
            for shipment in arrived:
                warehouse.receive_shipment(shipment['quantity'])


class DemandAgent(Agent):
    """Demand generation agent"""
    
    def __init__(self, unique_id, model, forecaster):
        super().__init__(unique_id, model)
        self.forecaster = forecaster
    
    def generate_demand(self):
        """Generate daily demand with variance"""
        recent_data = self.model.data_layer.historical_data.tail(30)
        predicted = self.forecaster.predict(recent_data, steps=1)[0]
        
        # Add realistic variance
        variance = predicted * 0.3
        actual = max(0, predicted + np.random.normal(0, variance))
        
        return int(actual)
    
    def step(self):
        """Agent step function"""
        demand = self.generate_demand()
        self.model.daily_demand = demand
