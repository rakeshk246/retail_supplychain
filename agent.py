# agents.py
# ==============================================================================
# Defines agents compatible with SupplyChainModel (v2)
# SupplierAgent, WarehouseAgent, LogisticsAgent, DemandAgent
# ==============================================================================

from mesa import Agent
import random

# ----------------------------------------------------------------------
# Supplier Agent
# ----------------------------------------------------------------------
class SupplierAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.status = "active"

    def process_order(self, quantity):
        """Process incoming order if supplier is active."""
        if self.status == "disrupted":
            self.model.log_event("Supplier", "Unable to fulfill order due to disruption")
            return 0
        
        shipped = quantity
        self.model.log_event("Supplier", f"Processed order for {shipped} units")
        return shipped


# ----------------------------------------------------------------------
# Warehouse Agent
# ----------------------------------------------------------------------
class WarehouseAgent(Agent):
    def __init__(self, unique_id, model, initial_inventory=500, reorder_point=200, reorder_qty=300):
        super().__init__(unique_id, model)
        self.inventory = initial_inventory
        self.reorder_point = reorder_point
        self.reorder_qty = reorder_qty

    def fulfill_demand(self, demand):
        """Fulfill daily demand from inventory."""
        fulfilled = min(self.inventory, demand)
        self.inventory -= fulfilled
        if fulfilled < demand:
            self.model.stockouts += 1
            self.model.log_event("Warehouse", "Stockout occurred")
        return fulfilled

    def check_reorder(self, predicted_demand):
        """Check if inventory is below reorder point and reorder."""
        if self.inventory < self.reorder_point:
            order_qty = self.reorder_qty
            self.model.log_event("Warehouse", f"Reordering {order_qty} units")
            return order_qty
        return 0


# ----------------------------------------------------------------------
# Logistics Agent
# ----------------------------------------------------------------------
class LogisticsAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.status = "active"
        self.shipments = []

    def schedule_shipment(self, quantity):
        """Schedule shipment for delivery."""
        if self.status == "disrupted":
            self.model.log_event("Logistics", "Shipment delayed due to disruption")
            return
        self.shipments.append(quantity)
        self.model.log_event("Logistics", f"Shipment of {quantity} units scheduled")

    def step(self):
        """Deliver scheduled shipments."""
        if self.status == "disrupted":
            self.model.log_event("Logistics", "No delivery — system disrupted")
            return
        if self.shipments:
            delivered = sum(self.shipments)
            self.model.warehouse.inventory += delivered
            self.model.log_event("Logistics", f"Delivered {delivered} units to warehouse")
            self.shipments.clear()


# ----------------------------------------------------------------------
# Demand Agent
# ----------------------------------------------------------------------
class DemandAgent(Agent):
    def __init__(self, unique_id, model, forecaster):
        super().__init__(unique_id, model)
        self.forecaster = forecaster

    def step(self):
        """Generate daily demand using the forecasting model."""
        demand = int(self.forecaster.predict_next())
        self.model.daily_demand = demand
        self.model.log_event("Demand", f"Predicted daily demand: {demand}")
