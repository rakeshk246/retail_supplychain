# scenarios_colorful.py
# ==============================================================================
# V3: Highly Relatable & Visual Scenarios
# Scenarios designed to instantly make sense to non-technical users
# ==============================================================================

class Scenario:
    def __init__(self, name, icon, description, total_days, highlights,
                 human_metrics, disruption_plan=None, demand_overrides=None):
        self.name = name
        self.icon = icon
        self.description = description
        self.total_days = total_days
        self.highlights = highlights
        self.human_metrics = human_metrics
        self.disruption_plan = disruption_plan or {}
        self.demand_overrides = demand_overrides or {}

    def apply_day(self, day, model, base_demand):
        if day in self.disruption_plan:
            dtype, duration = self.disruption_plan[day]
            model.inject_disruption(dtype, duration)
        
        multiplier = self.demand_overrides.get(day, 1.0)
        adjusted_demand = max(1, int(base_demand * multiplier))
        return adjusted_demand


def get_colorful_scenarios():
    """Return the 3 highly relatable and colorful scenarios."""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SCENARIO 1: The Viral TikTok Trend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    scenario1 = Scenario(
        name="The Viral TikTok Trend",
        icon="📱",
        description=(
            "An influencer just posted about your product, and it went viral overnight! "
            "Demand absolutely explodes by 500% for 3 days before cooling off. "
            "Watch how the AI agents frantically coordinate via the Agent Comms chatroom "
            "to place massive emergency restocks before the warehouse runs dry."
        ),
        total_days=15,
        highlights=[
            "🚀 Immediate 500% surge in customer demand",
            "💬 Intense communication between Demand & Warehouse agents",
            "⚡ Lightning-fast emergency orders sent to Supplier",
            "🛡️ Prevents days of stockouts compared to human teams",
        ],
        human_metrics={
            "Response Time": "1-2 days (wait for trend report)",
            "Stockout Duration": "4-6 days",
            "Fill Rate": "~40%",
        },
        demand_overrides={
            # Normal days 1-3
            4: 2.0, 5: 5.0, 6: 4.5, 7: 3.0, # The viral spike
            8: 1.5, 9: 1.2, # The tail end
        },
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SCENARIO 2: The Major Port Strike
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    scenario2 = Scenario(
        name="The Major Port Strike",
        icon="🛑",
        description=(
            "Dockworkers at the main port have gone on strike! The logistics network "
            "is completely paralyzed for 7 full days. No new shipments can arrive. "
            "Watch how the AI handles a true crisis without panicking, carefully rationing "
            "what's left while desperately trying to route around the blockage."
        ),
        total_days=20,
        highlights=[
            "🚢 Logistics agent reports total gridlock (CRISIS path)",
            "🧠 Warehouse relies entirely on existing buffer stock",
            "📉 Fill rate struggles, but survival is prioritized",
            "🤖 AI avoids canceling orders, queues them for recovery",
        ],
        human_metrics={
            "Response Time": "Immediate panic",
            "Stockout Duration": "Guaranteed during strike",
            "Fill Rate": "Very low",
        },
        disruption_plan={
            5: ('logistics', 7),   # Logistics goes down for a full 7 days!
        },
        demand_overrides={
            # Keep demand mostly normal to isolate the disruption effect
            5: 1.1, 6: 1.0, 7: 1.2,
        },
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SCENARIO 3: The "Bullwhip Effect" Trap
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    scenario3 = Scenario(
        name="The Bullwhip Effect Trap",
        icon="🎢",
        description=(
            "A classic supply chain trap: customer demand is wildly volatile, swinging "
            "up 300% one day, then dropping 80% the next. Human teams usually overreact, "
            "ordering too much and flooding the warehouse. Watch the AI agents stay calm, "
            "communicate clearly, and average out the noise."
        ),
        total_days=15,
        highlights=[
            "📉 Artificial wild swings in daily demand",
            "🧘‍♂️ AI resists the urge to over-order",
            "🚫 Prevents massive overstocking at the end of the month",
            "🤝 Proof that perfect communication beats panic",
        ],
        human_metrics={
            "Response Time": "Constant over-reaction",
            "Stockout Duration": "Low, but inventory costs explode",
            "Final Inventory": "Massive overflow",
        },
        demand_overrides={
            2: 0.5, 3: 3.5, 4: 0.2, 5: 3.0, 
            6: 0.5, 7: 2.8, 8: 0.3, 9: 3.2,
            10: 0.8, 11: 2.5, 12: 0.4,
        },
    )

    return [scenario1, scenario2, scenario3]
