# scenarios.py
# ==============================================================================
# V2: Pre-built Test Case Scenarios
# Three carefully designed scenarios that demonstrate AI agent capabilities
# ==============================================================================

import numpy as np


class Scenario:
    """A pre-built test case scenario."""

    def __init__(self, name, icon, description, total_days, highlights,
                 human_metrics, disruption_plan=None, demand_overrides=None):
        self.name = name
        self.icon = icon
        self.description = description
        self.total_days = total_days
        self.highlights = highlights  # List of what this demonstrates
        self.human_metrics = human_metrics  # Estimated human performance for comparison
        self.disruption_plan = disruption_plan or {}  # {day: ('type', duration)}
        self.demand_overrides = demand_overrides or {}  # {day: multiplier}

    def apply_day(self, day, model, base_demand):
        """Apply scenario effects for a given day.

        Args:
            day: current simulation day (1-indexed)
            model: OrchestratedSupplyChainModel
            base_demand: the base demand from the forecaster

        Returns:
            int: adjusted demand for this day
        """
        # Apply disruptions
        if day in self.disruption_plan:
            dtype, duration = self.disruption_plan[day]
            model.inject_disruption(dtype, duration)

        # Apply demand overrides
        multiplier = self.demand_overrides.get(day, 1.0)
        adjusted_demand = max(1, int(base_demand * multiplier))
        return adjusted_demand


def get_scenarios():
    """Return the 3 pre-built test case scenarios."""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SCENARIO 1: Supplier Disruption + Demand Spike
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    scenario1 = Scenario(
        name="Supplier Disruption + Demand Spike",
        icon="🌪️",
        description=(
            "The worst-case scenario: a supplier goes offline for 4 days "
            "while customer demand simultaneously spikes to 2.5× normal levels. "
            "This tests how AI agents coordinate under maximum pressure — "
            "communicating via the Message Bus, switching to CRISIS workflow, "
            "and making rapid decisions that a human team would take hours to reach."
        ),
        total_days=15,
        highlights=[
            "📡 Agent-to-agent communication via Message Bus",
            "🔀 LangGraph routes to CRISIS path automatically",
            "🧠 LLM reasons about partial fulfillment strategies",
            "⚡ AI resolves in seconds vs. human hours",
        ],
        human_metrics={
            "Response Time": "8-12 hours",
            "Stockout Duration": "3-4 days",
            "Fill Rate": "~55%",
            "Recovery Time": "5-6 days",
            "Learning": "No systematic learning",
        },
        disruption_plan={
            6: ('supplier', 4),   # Supplier goes down on day 6 for 4 days
        },
        demand_overrides={
            # Normal demand days 1-5
            6: 2.5, 7: 2.8, 8: 2.5, 9: 2.0,  # Demand spike during disruption
            10: 1.5, 11: 1.2,  # Gradual return to normal
            # Normal demand days 12-15
        },
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SCENARIO 2: Rising Demand Trend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Demand grows ~12% per day (festival/holiday season)
    rising_overrides = {}
    for d in range(1, 16):
        rising_overrides[d] = 1.0 + (d - 1) * 0.12  # 1.0, 1.12, 1.24, ..., 2.68

    scenario2 = Scenario(
        name="Rising Demand Trend",
        icon="📈",
        description=(
            "A gradual demand increase over 15 days simulating a holiday season — "
            "demand grows ~12% each day. No sudden disruptions, but the rising tide "
            "will cause stockouts if agents don't proactively increase orders. "
            "This tests whether AI agents can detect trends early and act before "
            "problems occur — something humans typically notice too late."
        ),
        total_days=15,
        highlights=[
            "📈 LSTM forecasting detects the rising trend early",
            "📚 Memory recalls similar past seasonal patterns",
            "🛡️ Proactive ordering prevents stockout",
            "🧠 AI spots the trend on Day 2; humans notice on Day 5+",
        ],
        human_metrics={
            "Response Time": "4-5 days (notices trend late)",
            "Stockout Duration": "2-3 days",
            "Fill Rate": "~72%",
            "Recovery Time": "3-4 days",
            "Learning": "Relies on individual memory",
        },
        demand_overrides=rising_overrides,
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SCENARIO 3: Repeat Disruption (Memory Learning)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    scenario3 = Scenario(
        name="Repeat Disruption — Memory Learning",
        icon="🧠",
        description=(
            "Two identical supplier disruptions: one on Day 5 and another on Day 15. "
            "The key question: does the AI handle the second disruption better than the first? "
            "Thanks to ChromaDB vector memory, the agents remember what happened during "
            "Disruption #1 and use that experience to respond faster and smarter to "
            "Disruption #2. Rule-based systems make the same mistake twice."
        ),
        total_days=20,
        highlights=[
            "📚 ChromaDB memory stores Disruption #1 experience",
            "🧠 Agent recalls past episode during Disruption #2",
            "📊 Measurably better response the second time",
            "🔄 Proves AI agents learn — rules don't",
        ],
        human_metrics={
            "Response Time": "Same both times (8+ hours)",
            "Stockout Duration": "Same both times (2-3 days)",
            "Fill Rate": "~65% (no improvement)",
            "Recovery Time": "Same both times (4-5 days)",
            "Learning": "No systematic learning",
        },
        disruption_plan={
            5: ('supplier', 3),   # First disruption: Day 5 for 3 days
            15: ('supplier', 3),  # Second disruption: Day 15 for 3 days
        },
        demand_overrides={
            # Mild demand increase during disruptions to add pressure
            5: 1.3, 6: 1.4, 7: 1.3,
            15: 1.3, 16: 1.4, 17: 1.3,
        },
    )

    return [scenario1, scenario2, scenario3]
