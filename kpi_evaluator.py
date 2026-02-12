class KPIEvaluator:
    """Evaluate supply chain performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_demand': 0,
            'fulfilled_demand': 0,
            'stockouts': 0,
            'total_days': 0,
            'inventory_sum': 0
        }
    
    def update(self, demand, fulfilled, inventory, stockout):
        """Update metrics"""
        self.metrics['total_demand'] += demand
        self.metrics['fulfilled_demand'] += fulfilled
        if stockout:
            self.metrics['stockouts'] += 1
        self.metrics['total_days'] += 1
        self.metrics['inventory_sum'] += inventory
    
    def calculate_kpis(self):
        """Calculate all KPIs"""
        if self.metrics['total_days'] == 0:
            return {}
        
        fill_rate = (self.metrics['fulfilled_demand'] / 
                    max(self.metrics['total_demand'], 1)) * 100
        
        stockout_rate = (self.metrics['stockouts'] / 
                        self.metrics['total_days']) * 100
        
        avg_inventory = (self.metrics['inventory_sum'] / 
                        self.metrics['total_days'])
        
        resilience_index = max(0, 100 - stockout_rate * 2)
        
        customer_satisfaction = fill_rate * 0.7 + resilience_index * 0.3
        
        return {
            'Fill Rate (%)': round(fill_rate, 2),
            'Stock-out Rate (%)': round(stockout_rate, 2),
            'Avg Inventory': round(avg_inventory, 2),
            'Resilience Index': round(resilience_index, 2),
            'Customer Satisfaction': round(customer_satisfaction, 2)
        }
