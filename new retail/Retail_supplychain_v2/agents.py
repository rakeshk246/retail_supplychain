# agents.py
# ==============================================================================
# Unified Agent Module — Phase 1 (Foundation)
# All Mesa agents for the supply chain simulation
# Compatible with Mesa 3.0+ (Agent takes only model, unique_id auto-assigned)
# Bug fixes applied: Lead time simulation, disruption handling, proper imports
# ==============================================================================

from mesa import Agent
import random
import numpy as np


# ----------------------------------------------------------------------
# Supplier Agent
# ----------------------------------------------------------------------
class SupplierAgent(Agent):
    """Supplier agent that processes orders with reliability and capacity constraints."""

    def __init__(self, model, reliability=0.95, capacity=500, lead_time=2):
        super().__init__(model)
        self.status = "active"
        self.reliability = reliability
        self.capacity = capacity
        self.lead_time = lead_time
        self.total_orders = 0
        self.fulfilled_orders = 0

    def process_order(self, quantity):
        """Process incoming order if supplier is active.
        
        Returns:
            dict with order details if successful, 0 if disrupted
        """
        if self.status == "disrupted":
            self.model.log_event("Supplier", "Unable to fulfill order due to disruption")
            return 0

        # Apply capacity constraint
        actual_qty = min(quantity, self.capacity)
        if actual_qty < quantity:
            self.model.log_event(
                "Supplier",
                f"Partial fulfillment: {actual_qty}/{quantity} units (capacity limit)"
            )
        else:
            self.model.log_event("Supplier", f"Processed order for {actual_qty} units")

        # Apply reliability — random chance of partial delivery
        if random.random() > self.reliability:
            actual_qty = int(actual_qty * random.uniform(0.7, 0.95))
            self.model.log_event(
                "Supplier",
                f"Reliability event: only {actual_qty} units shipped"
            )

        self.total_orders += 1
        self.fulfilled_orders += 1

        return {
            'quantity': actual_qty,
            'lead_time': self.lead_time
        }

    def step(self):
        """Agent step — supplier doesn't act autonomously in rule-based mode."""
        pass


# ----------------------------------------------------------------------
# Warehouse Agent
# ----------------------------------------------------------------------
class WarehouseAgent(Agent):
    """Warehouse agent managing inventory with reorder logic."""

    def __init__(self, model, initial_inventory=500,
                 reorder_point=200, reorder_qty=300, max_capacity=1000):
        super().__init__(model)
        self.inventory = initial_inventory
        self.reorder_point = reorder_point
        self.reorder_qty = reorder_qty
        self.max_capacity = max_capacity
        self.target_stock = 500

    def fulfill_demand(self, demand):
        """Fulfill daily demand from inventory.
        
        Returns:
            int — number of units actually fulfilled
        """
        fulfilled = min(self.inventory, demand)
        self.inventory -= fulfilled
        if fulfilled < demand:
            self.model.stockouts += 1
            shortfall = demand - fulfilled
            self.model.log_event(
                "Warehouse",
                f"STOCKOUT: fulfilled {fulfilled}/{demand} units (short {shortfall})"
            )
        else:
            self.model.log_event(
                "Warehouse",
                f"Fulfilled demand: {fulfilled} units | Remaining: {self.inventory}"
            )
        return fulfilled

    def check_reorder(self, predicted_demand):
        """Check if inventory is below reorder point and calculate order quantity.
        
        Args:
            predicted_demand: forecasted demand for upcoming period
            
        Returns:
            int — quantity to order (0 if no reorder needed)
        """
        if self.inventory < self.reorder_point:
            # Dynamic order quantity based on predicted demand
            buffer = predicted_demand * 1.3  # 30% safety buffer
            order_qty = max(
                self.reorder_qty,
                int(self.target_stock - self.inventory + buffer)
            )
            # Don't exceed warehouse capacity
            order_qty = min(order_qty, self.max_capacity - self.inventory)
            self.model.log_event(
                "Warehouse",
                f"Reorder triggered: inventory={self.inventory} < reorder_point={self.reorder_point} | "
                f"Ordering {order_qty} units"
            )
            return order_qty
        return 0

    def receive_shipment(self, quantity):
        """Receive incoming shipment into warehouse."""
        space_available = self.max_capacity - self.inventory
        accepted = min(quantity, space_available)
        self.inventory += accepted
        if accepted < quantity:
            self.model.log_event(
                "Warehouse",
                f"Received {accepted}/{quantity} units (capacity limit) | Stock: {self.inventory}"
            )
        else:
            self.model.log_event(
                "Warehouse",
                f"Received {accepted} units | Stock: {self.inventory}"
            )

    def step(self):
        """Agent step — warehouse doesn't act autonomously in rule-based mode."""
        pass


# ----------------------------------------------------------------------
# Logistics Agent — BUG 4 FIXED: Proper lead time simulation
# ----------------------------------------------------------------------
class LogisticsAgent(Agent):
    """Logistics agent with proper lead time simulation.
    
    Shipments are scheduled with an arrival_day. They are only delivered
    to the warehouse when current_day >= arrival_day. This simulates
    realistic transit delays.
    """

    def __init__(self, model, default_lead_time=3):
        super().__init__(model)
        self.status = "active"
        self.default_lead_time = default_lead_time
        self.shipments = []  # List of {'quantity': int, 'arrival_day': int}
        self.total_shipments = 0
        self.delivered_shipments = 0

    def schedule_shipment(self, quantity, lead_time=None):
        """Schedule a shipment with proper lead time tracking.
        
        Args:
            quantity: number of units to ship (can be int or dict with 'quantity' key)
            lead_time: days until arrival (uses default if None)
        """
        # Handle both dict (from supplier) and int inputs
        if isinstance(quantity, dict):
            lead_time = quantity.get('lead_time', self.default_lead_time)
            quantity = quantity.get('quantity', 0)

        if lead_time is None:
            lead_time = self.default_lead_time

        if self.status == "disrupted":
            lead_time += 2  # Disruption adds delay
            self.model.log_event(
                "Logistics",
                f"Shipment delayed due to disruption — extra 2 days added"
            )

        arrival_day = self.model.current_day + lead_time
        self.shipments.append({
            'quantity': quantity,
            'arrival_day': arrival_day,
            'scheduled_day': self.model.current_day
        })
        self.total_shipments += 1
        self.model.log_event(
            "Logistics",
            f"Shipment of {quantity} units scheduled | "
            f"Arrives day {arrival_day} (lead time: {lead_time} days)"
        )

    def step(self):
        """Process shipments — deliver arrived, keep pending."""
        arrived = [s for s in self.shipments
                   if s['arrival_day'] <= self.model.current_day]
        pending = [s for s in self.shipments
                   if s['arrival_day'] > self.model.current_day]
        self.shipments = pending

        for shipment in arrived:
            self.model.warehouse.receive_shipment(shipment['quantity'])
            self.delivered_shipments += 1
            transit_days = shipment['arrival_day'] - shipment['scheduled_day']
            self.model.log_event(
                "Logistics",
                f"Delivered {shipment['quantity']} units to warehouse "
                f"(transit: {transit_days} days)"
            )

        if pending:
            self.model.log_event(
                "Logistics",
                f"{len(pending)} shipment(s) still in transit"
            )


# ----------------------------------------------------------------------
# Demand Agent
# ----------------------------------------------------------------------
class DemandAgent(Agent):
    """Demand agent that generates daily demand using the forecasting model."""

    def __init__(self, model, forecaster):
        super().__init__(model)
        self.forecaster = forecaster

    def step(self):
        """Generate daily demand using the forecasting model."""
        try:
            predicted = self.forecaster.predict_next()
            # Add realistic variance (±20%)
            variance = predicted * 0.2
            demand = max(1, int(predicted + np.random.normal(0, variance)))
        except Exception:
            # Fallback to random demand if forecaster fails
            demand = random.randint(50, 150)
        
        self.model.daily_demand = demand
        self.model.log_event("Demand", f"Daily demand: {demand} units")
