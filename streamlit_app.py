import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json

from data_layer import DataLayer
from forecasting_module import DemandForecaster
from orchestrator import OrchestratedSupplyChainModel
from agentic_agents import LLMEngine, GroqRateLimitError
from explainability import ExplainabilityEngine
from message_bus import MessageBus
from kpi_evaluator import KPIEvaluator


# ==============================================================================
# HELPERS
# ==============================================================================

def init_model(dl, fc, mode):
    LLMEngine.reset()
    MessageBus.reset()
    ExplainabilityEngine.reset()
    return OrchestratedSupplyChainModel(fc, dl, agent_mode=mode)


def run_one_step(model, kpi_eval, sim_data):
    """Run one step. Raises GroqRateLimitError if rate limited in agentic mode."""
    inv_before = model.warehouse.inventory
    supplier_before = model.supplier.status
    logistics_before = model.logistics.status
    pending_before = len(model.logistics.shipments)

    # Get LSTM base prediction before step
    try:
        lstm_prediction = model.forecaster.predict_next() if hasattr(model, 'forecaster') else None
    except Exception:
        lstm_prediction = None

    # Try to map simulation day to source dataset date
    source_date = None
    source_demand = None
    dl = model.data_layer if hasattr(model, 'data_layer') else None
    if dl and dl.historical_data is not None:
        hist = dl.historical_data
        # Map current day to a date in the dataset (use modulo for cycling)
        idx = (model.current_day) % len(hist)
        row = hist.iloc[idx]
        source_date = str(row['ds'])[:10] if 'ds' in hist.columns else None
        source_demand = float(row['y']) if 'y' in hist.columns else None

    # This will raise GroqRateLimitError if rate limit hit in agentic mode
    model.step()

    fulfilled = min(model.daily_demand, inv_before)
    stockout = fulfilled < model.daily_demand

    kpi_eval.update(
        model.daily_demand, fulfilled, model.warehouse.inventory,
        stockout, day=model.current_day,
        disruption_active=bool(model.disruption_schedule))

    # Get XAI decisions for this day
    decisions = model.xai.get_decision_chain(model.current_day)
    why_texts = []
    agent_decisions = []  # Detailed per-agent decisions
    for d in decisions:
        w = d.get('why', {}).get('summary', '')
        action = d.get('action', '')
        agent = d.get('agent', '')
        dtype = d.get('decision_type', '')
        factors = d.get('why', {}).get('contributing_factors', [])
        is_llm = d.get('is_llm_decision', False)

        agent_decisions.append({
            'agent': agent,
            'type': dtype,
            'action': action,
            'why': w,
            'factors': factors,
            'is_llm': is_llm,
        })
        if w and dtype not in ('forecast', 'fulfill_demand', 'delivery'):
            why_texts.append(f"{agent}: {w}")

    # Get agent reasoning
    agent_reasoning = model.get_agent_reasoning() if hasattr(model, 'get_agent_reasoning') else {}

    record = {
        'day': model.current_day,
        'inv_before': inv_before,
        'inv_after': model.warehouse.inventory,
        'demand': model.daily_demand,
        'fulfilled': fulfilled,
        'stockout': stockout,
        'path': model._last_workflow_path,
        'supplier': model.supplier.status,
        'supplier_before': supplier_before,
        'logistics': model.logistics.status,
        'logistics_before': logistics_before,
        'pending': len(model.logistics.shipments),
        'pending_before': pending_before,
        'why_texts': why_texts,
        # New transparency fields
        'source_date': source_date,
        'source_demand': source_demand,
        'lstm_prediction': lstm_prediction,
        'agent_decisions': agent_decisions,
        'agent_reasoning': agent_reasoning,
        'data_source': dl.data_source if dl else 'unknown',
        'mode': model.agent_mode if hasattr(model, 'agent_mode') else 'unknown',
    }
    sim_data.append(record)
    return record


def compute_score(kpis):
    fill = float(kpis.get('Fill Rate (%)', 100))
    sat = float(kpis.get('Customer Satisfaction', 100))
    return int((fill * 0.6 + sat * 0.4))


# ==============================================================================
# SUPPLY CHAIN MAP (rendered as HTML component)
# ==============================================================================

def render_map(model):
    s = model.supplier.status
    l = model.logistics.status
    inv = model.warehouse.inventory
    dem = model.daily_demand
    pend = len(model.logistics.shipments)
    sc = "#22c55e" if s == "active" else "#ef4444"
    lc = "#22c55e" if l == "active" else "#ef4444"
    pct = min(inv / 1000, 1.0) * 100
    bc = "#22c55e" if inv > 200 else "#eab308" if inv > 50 else "#ef4444"
    sa = "glow" if s == "active" and pend > 0 else ""
    la = "glow" if l == "active" and pend > 0 else ""

    html = f"""<!DOCTYPE html><html><head><style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#0f172a; font-family:system-ui,sans-serif; }}
    @keyframes glow {{ 0%,100%{{opacity:.3}} 50%{{opacity:1}} }}
    .row {{ display:flex; align-items:center; justify-content:center; gap:8px; padding:16px 8px; }}
    .node {{ text-align:center; padding:14px 12px; border-radius:12px; min-width:120px; }}
    .icon {{ font-size:30px; }}
    .name {{ color:#e2e8f0; font-weight:700; font-size:13px; margin:4px 0; }}
    .info {{ color:#94a3b8; font-size:11px; }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:6px; font-size:10px;
              font-weight:700; margin-top:4px; }}
    .arrow {{ font-size:22px; color:#475569; }}
    .arrow.glow {{ animation:glow 1.2s ease-in-out infinite; color:#22c55e; }}
    .bar {{ width:100%; height:6px; background:#1e293b; border-radius:3px; margin-top:6px; }}
    .fill {{ height:100%; border-radius:3px; }}
    </style></head><body>
    <div class="row">
      <div class="node" style="background:linear-gradient(135deg,#1e3a5f,#0f2440);border:2px solid {sc}">
        <div class="icon">🏭</div><div class="name">Supplier</div>
        <div class="info">Capacity: {getattr(model.supplier,'capacity',500)}</div>
        <div class="badge" style="background:{sc}22;color:{sc}">{s.upper()}</div>
      </div>
      <div class="arrow {sa}">📦 ➡️</div>
      <div class="node" style="background:linear-gradient(135deg,#2d1f4e,#1a1333);border:2px solid {lc}">
        <div class="icon">🚛</div><div class="name">Logistics</div>
        <div class="info">{pend} in transit</div>
        <div class="badge" style="background:{lc}22;color:{lc}">{l.upper()}</div>
      </div>
      <div class="arrow {la}">📦 ➡️</div>
      <div class="node" style="background:linear-gradient(135deg,#1a2e1a,#0f1f0f);border:2px solid {bc}">
        <div class="icon">🏬</div><div class="name">Warehouse</div>
        <div class="info">{inv} units</div>
        <div class="bar"><div class="fill" style="width:{pct:.0f}%;background:{bc}"></div></div>
      </div>
      <div class="arrow glow">🛒 ➡️</div>
      <div class="node" style="background:linear-gradient(135deg,#3d1f1f,#2a1010);border:2px solid #8b5cf6">
        <div class="icon">🛍️</div><div class="name">Customers</div>
        <div class="info">Want: {dem} units</div>
        <div class="badge" style="background:#8b5cf622;color:#8b5cf6">BUYING</div>
      </div>
    </div>
    </body></html>"""
    components.html(html, height=150, scrolling=False)


# ==============================================================================
# AGENT CARDS WITH THINKING BUBBLES
# ==============================================================================

def render_agent_cards(model, record):
    day = model.current_day
    decisions = model.xai.get_decision_chain(day)

    # Get thoughts per agent
    thoughts = {}
    for d in decisions:
        agent = d['agent']
        why = d.get('why', {}).get('summary', '')
        if why:
            thoughts[agent] = why

    cards = [
        ("🏭", "Supplier", model.supplier.status,
         f"Cap: {getattr(model.supplier, 'capacity', 500)}",
         thoughts.get('Supplier', 'Waiting for orders...')),
        ("🚛", "Logistics", model.logistics.status,
         f"{len(model.logistics.shipments)} in transit",
         thoughts.get('Logistics', 'Routes clear.')),
        ("🏬", "Warehouse", "ok" if model.warehouse.inventory > 200 else "low",
         f"{model.warehouse.inventory} units",
         thoughts.get('Warehouse', 'Monitoring stock levels.')),
        ("🛍️", "Demand", "active",
         f"{model.daily_demand} units",
         thoughts.get('Demand', 'Analyzing buying patterns.')),
    ]

    cols = st.columns(4)
    for col, (icon, name, status, info, thought) in zip(cols, cards):
        with col:
            color = "🟢" if status == "active" or status == "ok" else "🔴" if status == "disrupted" else "🟡"
            st.markdown(f"### {icon} {name} {color}")
            st.caption(info)
            st.info(f'💭 *"{thought[:100]}"*')


# ==============================================================================
# INVENTORY GAUGE
# ==============================================================================

def render_gauge(inv):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=inv,
        title={'text': "Warehouse Stock", 'font': {'size': 16, 'color': '#e2e8f0'}},
        number={'font': {'size': 36, 'color': '#e2e8f0'}},
        gauge={
            'axis': {'range': [0, 1000], 'tickcolor': '#475569'},
            'bar': {'color': '#3b82f6'},
            'bgcolor': '#1e293b',
            'steps': [
                {'range': [0, 100], 'color': '#7f1d1d'},
                {'range': [100, 200], 'color': '#78350f'},
                {'range': [200, 1000], 'color': '#14532d'}
            ],
            'threshold': {
                'line': {'color': '#f59e0b', 'width': 3},
                'thickness': 0.8,
                'value': 200
            }
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': '#e2e8f0'})
    return fig


# ==============================================================================
# SHIPMENT TRACKER
# ==============================================================================

def render_shipments(model):
    shipments = model.logistics.shipments
    if not shipments:
        st.caption("No shipments in transit")
        return

    for i, s in enumerate(shipments):
        if isinstance(s, dict):
            qty = s.get('quantity', 0)
            arrival = s.get('arrival_day', 0)
        else:
            qty = getattr(s, 'quantity', 0)
            arrival = getattr(s, 'arrival_day', 0)

        days_left = max(0, arrival - model.current_day)
        total_days = getattr(model.supplier, 'lead_time', 3)
        progress = max(0, 1 - (days_left / max(total_days, 1)))

        bar_color = "#22c55e" if days_left <= 1 else "#3b82f6"
        st.markdown(
            f"📦 **{qty} units** — arrives Day {arrival} "
            f"({'tomorrow!' if days_left == 1 else f'{days_left} days'})")
        st.progress(min(progress, 1.0))


# ==============================================================================
# DAY NARRATIVE
# ==============================================================================

def render_narrative(record, model):
    d = record['day']
    dem = record['demand']
    ib = record['inv_before']
    ia = record['inv_after']
    ful = record['fulfilled']
    path = record['path']
    so = record['stockout']
    source_date = record.get('source_date', None)
    source_demand = record.get('source_demand', None)
    lstm_pred = record.get('lstm_prediction', None)
    data_source = record.get('data_source', 'unknown')
    mode = record.get('mode', 'unknown')
    agent_decisions = record.get('agent_decisions', [])
    agent_reasoning = record.get('agent_reasoning', {})

    # ===================== HEADER =====================
    pe = {"NORMAL": "🟢", "EMERGENCY": "🟡", "CRISIS": "🔴"}.get(path, "⚪")
    st.markdown(f"## 📅 Day {d} — {pe} {path} Path")

    # ===================== DATA SOURCE =====================
    st.markdown("### 📂 Where This Data Comes From")
    source_name = "Walmart M5 Dataset" if data_source == 'm5' else "Synthetic Data"
    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.metric("📊 Data Source", source_name)
    with cols[1]:
        if source_date:
            st.metric("📆 Dataset Date", source_date)
        else:
            st.metric("📆 Sim Day", f"Day {d}")
    with cols[2]:
        if source_demand is not None:
            st.metric("📈 Historical Demand", f"{source_demand:.0f} units")

    # ===================== SCENARIO TABLE =====================
    st.markdown("### 🎯 Day Scenario — What Did Each Agent See?")

    pred_text = f"{lstm_pred:.0f} units" if lstm_pred else "N/A"
    mode_text = {"agentic": "🤖 Agentic (LLM)", "rule_based": "📐 Rule-based", "hybrid": "🔀 Hybrid"}.get(mode, mode)

    scenario_md = f"""
    | Parameter | Value | Meaning |
    |-----------|-------|--------|
    | **Mode** | {mode_text} | How decisions are made |
    | **Starting Inventory** | 📦 **{ib} units** | Stock available at start of day |
    | **LSTM Forecast** | 🔮 {pred_text} | Model's prediction for today's demand |
    | **Actual Demand** | 🛒 **{dem} units** | What customers actually ordered |
    | **Fulfilled** | {'✅' if not so else '❌'} **{ful}/{dem} units** | How much we could ship |
    | **Ending Inventory** | 📦 **{ia} units** | Stock remaining after fulfillment |
    | **Supplier Status** | {'🟢 Active' if record.get('supplier_before') == 'active' else '🔴 Disrupted'} | Can we order new stock? |
    | **Logistics Status** | {'🟢 Active' if record.get('logistics_before') == 'active' else '🔴 Disrupted'} | Can shipments arrive? |
    | **Pending Shipments** | 🚚 {record.get('pending_before', 0)} → {record.get('pending', 0)} | Orders in transit |
    """
    st.markdown(scenario_md)

    # ===================== AGENT DECISION TIMELINE =====================
    if agent_decisions:
        st.markdown("### 🤖 Agent Decision Timeline — Step by Step")
        for i, ad in enumerate(agent_decisions):
            agent = ad.get('agent', '?')
            action = ad.get('action', 'N/A')
            why = ad.get('why', '')
            dtype = ad.get('type', '')
            is_llm = ad.get('is_llm', False)
            factors = ad.get('factors', [])

            emoji = {'Demand': '📈', 'Warehouse': '📦', 'Supplier': '🏭', 'Logistics': '🚚'}.get(agent, '🤖')
            source_badge = '`🧠 LLM`' if is_llm else '`📐 Rules`'

            with st.expander(f"Step {i+1}: {emoji} **{agent}** — {action}", expanded=(i < 2)):
                st.markdown(f"**Decision Type:** {dtype}")
                st.markdown(f"**Decision Source:** {source_badge}")
                if why:
                    st.markdown(f"**Why:** {why}")
                if factors:
                    st.markdown("**Contributing Factors:**")
                    for f in factors:
                        st.markdown(f"- {f}")

    # ===================== AGENT REASONING (LLM) =====================
    if agent_reasoning:
        st.markdown("### 💭 Raw LLM Reasoning")
        for agent_name, reasoning in agent_reasoning.items():
            with st.expander(f"🧠 {agent_name}'s thought process"):
                st.code(reasoning[:500], language='text')

    # ===================== OUTCOME SUMMARY =====================
    st.markdown("### 📊 Outcome")
    if so:
        short = dem - ful
        st.error(f"❌ **STOCKOUT!** Customers wanted {dem} units but we only had {ib}. "
                 f"Shipped {ful}, **{short} orders unfulfilled.** Lost revenue: ~${short * 15:.0f}")
    else:
        st.success(f"✅ **All orders fulfilled!** Shipped {ful}/{dem} units. Stock: {ib} → {ia}")

    # Health
    if ia <= 0:
        st.error("🔴 **WAREHOUSE EMPTY!** No stock left!")
    elif ia < 100:
        days_left = ia / max(dem, 1)
        st.warning(f"🟡 **Low stock:** {ia} units (~{days_left:.1f} days supply)")

    # Disruptions
    for at, ed in model.disruption_schedule.items():
        rem = ed - d
        if rem > 0:
            st.warning(f"⚠️ **{at.title()} DISRUPTED** — recovers in {rem} day(s)")


# ==============================================================================
# WORKFLOW VISUALIZATION
# ==============================================================================

def render_workflow(model, sim_data):
    st.header("🔀 How LangGraph Routes Each Day")

    # Explain the workflow
    st.markdown("""
    **LangGraph** is a state machine that checks conditions and picks the best path each day.
    Instead of always doing the same thing, the AI **adapts** based on the situation:
    """)

    # Interactive workflow diagram
    workflow_html = """<!DOCTYPE html><html><head><style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background: #0f172a; font-family: system-ui, sans-serif; color: #e2e8f0; padding: 20px; }
    .flow { display: flex; flex-direction: column; align-items: center; gap: 8px; }
    .node { padding: 10px 24px; border-radius: 10px; font-weight: 600; font-size: 13px;
            text-align: center; min-width: 180px; }
    .start { background: #1e3a5f; border: 2px solid #3b82f6; }
    .check { background: #3d2d00; border: 2px solid #eab308; border-radius: 50%; 
             width: 160px; height: 60px; display: flex; align-items: center; justify-content: center; }
    .paths { display: flex; gap: 20px; justify-content: center; align-items: flex-start; }
    .path-box { text-align: center; padding: 12px; border-radius: 10px; min-width: 150px; }
    .normal { background: #14532d; border: 2px solid #22c55e; }
    .emergency { background: #78350f; border: 2px solid #eab308; }
    .crisis { background: #7f1d1d; border: 2px solid #ef4444; }
    .end { background: #1e293b; border: 2px solid #64748b; }
    .arrow { color: #64748b; font-size: 18px; }
    .label { font-size: 11px; color: #94a3b8; margin-top: 4px; }
    </style></head><body>
    <div class="flow">
      <div class="node start">1. Forecast Demand</div>
      <div class="arrow">⬇️</div>
      <div class="node start">2. Fulfill Customer Orders</div>
      <div class="arrow">⬇️</div>
      <div class="check">Check Inventory?</div>
      <div class="arrow">⬇️ ⬇️ ⬇️</div>
      <div class="paths">
        <div>
          <div class="path-box normal">🟢 NORMAL<div class="label">Stock > 200<br>Standard reorder</div></div>
        </div>
        <div>
          <div class="path-box emergency">🟡 EMERGENCY<div class="label">Stock < 200<br>Fast big reorder</div></div>
        </div>
        <div>
          <div class="path-box crisis">🔴 CRISIS<div class="label">Stock = 0 or disrupted<br>Maximum priority</div></div>
        </div>
      </div>
      <div class="arrow">⬇️</div>
      <div class="node start">4. Supplier Ships Order</div>
      <div class="arrow">⬇️</div>
      <div class="node start">5. Logistics Delivers</div>
      <div class="arrow">⬇️</div>
      <div class="node end">6. Record Metrics + XAI</div>
    </div>
    </body></html>"""
    components.html(workflow_html, height=520, scrolling=False)

    if not sim_data:
        return

    df = pd.DataFrame(sim_data)

    # Path timeline chart
    st.subheader("Path Chosen Each Day")
    path_map = {'NORMAL': 1, 'EMERGENCY': 2, 'CRISIS': 3}
    path_colors = {'NORMAL': '#22c55e', 'EMERGENCY': '#eab308', 'CRISIS': '#ef4444'}
    colors = [path_colors.get(p, '#64748b') for p in df['path']]

    fig = go.Figure(data=[go.Bar(
        x=df['day'], y=[path_map.get(p, 0) for p in df['path']],
        marker_color=colors, text=df['path'], textposition='auto')])
    fig.update_layout(height=200, template='plotly_dark',
        yaxis=dict(tickvals=[1, 2, 3], ticktext=['Normal', 'Emergency', 'Crisis']),
        xaxis_title="Day", margin=dict(l=20, r=20, t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Path distribution
    c1, c2, c3 = st.columns(3)
    counts = df['path'].value_counts()
    with c1:
        n = counts.get('NORMAL', 0)
        st.metric("🟢 Normal Days", n)
    with c2:
        e = counts.get('EMERGENCY', 0)
        st.metric("🟡 Emergency Days", e)
    with c3:
        cr = counts.get('CRISIS', 0)
        st.metric("🔴 Crisis Days", cr)


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.set_page_config(page_title="AI Supply Chain", page_icon="🤖", layout="wide")

    # Init
    if 'init' not in st.session_state:
        st.session_state.init = True
        st.session_state.dl = DataLayer()
        st.session_state.hist = st.session_state.dl.load_real_data()
        st.session_state.fc = DemandForecaster(method='lstm')
        with st.spinner("🧠 Training AI forecaster on Walmart M5 data..."):
            st.session_state.fc.train(st.session_state.hist, epochs=30)
        st.session_state.mode = 'agentic'
        st.session_state.model = init_model(st.session_state.dl, st.session_state.fc, 'agentic')
        st.session_state.kpi = KPIEvaluator()
        st.session_state.data = []
        st.session_state.rec = None

    model = st.session_state.model

    # Check LLM availability for agentic mode
    llm = LLMEngine.get_instance()
    if st.session_state.mode == 'agentic' and not llm.is_available:
        if llm.rate_limited:
            st.error(f"🛑 **GROQ RATE LIMIT EXCEEDED!** Simulation stopped. {llm.rate_limit_message}")
            st.stop()
        elif not llm.llm:
            st.warning("⚠️ **Agentic mode requires Groq API.** Set `GROQ_API_KEY` in `.env` file, or switch to `rule_based` mode in the sidebar.")

    # =================== HEADER ===================
    st.title("🤖 Agentic AI Supply Chain: California Grocery Focus")
    st.markdown("*Managing high-volume bottled water distribution for Walmart CA_1 using real-time Open-Meteo weather intelligence and DuckDuckGo news data.*")

    # Score + top metrics
    kpis = st.session_state.kpi.calculate_kpis()
    score = compute_score(kpis)
    hc = st.columns([1, 1, 1, 1, 1])
    with hc[0]:
        st.metric("🏆 Score", f"{score}/100")
    with hc[1]:
        st.metric("📅 Day", model.current_day)
    with hc[2]:
        st.metric("📦 Stock", model.warehouse.inventory)
    with hc[3]:
        pe = {"NORMAL": "🟢", "EMERGENCY": "🟡", "CRISIS": "🔴"}.get(model._last_workflow_path, "⚪")
        st.metric("Path", f"{pe} {model._last_workflow_path or '—'}")
    with hc[4]:
        st.metric("📈 Fill Rate", f"{kpis.get('Fill Rate (%)', 100)}%")

    # =================== SUPPLY CHAIN MAP ===================
    render_map(model)

    # =================== BUTTONS ===================
    bc = st.columns(6)
    with bc[0]:
        next_day = st.button("▶️ Next Day", use_container_width=True, type="primary")
    with bc[1]:
        run_10 = st.button("⏩ Run 10 Days", use_container_width=True)
    with bc[2]:
        hurricane = st.button("🌪️ Hurricane!", use_container_width=True)
    with bc[3]:
        road = st.button("🚧 Road Block!", use_container_width=True)
    with bc[4]:
        spike = st.button("📈 Demand Spike!", use_container_width=True)
    with bc[5]:
        reset = st.button("🔄 Reset", use_container_width=True)

    # Handle buttons
    try:
        if next_day:
            st.session_state.rec = run_one_step(model, st.session_state.kpi, st.session_state.data)
            st.rerun()
        if run_10:
            for _ in range(10):
                st.session_state.rec = run_one_step(model, st.session_state.kpi, st.session_state.data)
            st.rerun()
    except GroqRateLimitError as e:
        st.session_state['rate_limited'] = True
        st.session_state['rate_limit_msg'] = str(e)

    if hurricane:
        model.inject_disruption('supplier', 4)
        st.rerun()
    if road:
        model.inject_disruption('logistics', 3)
        st.rerun()
    if spike:
        model.warehouse.inventory = max(0, model.warehouse.inventory - 300)
        st.rerun()
    if reset:
        st.session_state.model = init_model(st.session_state.dl, st.session_state.fc, st.session_state.mode)
        st.session_state.kpi = KPIEvaluator()
        st.session_state.data = []
        st.session_state.rec = None
        st.session_state.pop('rate_limited', None)
        st.session_state.pop('rate_limit_msg', None)
        st.rerun()


    # HITL Approval Check
    if hasattr(model, 'hitl_pending') and model.hitl_pending:
        pending = model.hitl_pending
        st.warning(f"⚠️ **APPROVAL REQUIRED:** AI wants to order **{pending['qty']} units** (Reason: {pending.get('reason', 'Exceeds threshold')})")
        col1, col2 = st.columns(2)
        if col1.button("✅ Approve Order", type="primary"):
            model.approve_hitl_order()
            st.toast("Order Approved!")
            st.rerun()
        if col2.button("❌ Deny Order", type="secondary"):
            model.deny_hitl_order()
            st.toast("Order Denied!")
            st.rerun()
        st.stop()  # Stop execution until approved/denied

    # Show rate limit error if it occurred
    if st.session_state.get('rate_limited'):
        st.error(f"""
        ## 🛑 GROQ RATE LIMIT EXCEEDED!
        
        **The simulation has been STOPPED** because the Groq API rate limit was hit.
        
        The system does NOT fall back to rule-based mode — it strictly uses the LLM as requested.
        
        **Error:** {st.session_state.get('rate_limit_msg', 'Rate limit exceeded')}
        
        **What to do:**
        - ⏳ Wait for the rate limit to reset (usually resets daily)
        - 🔄 Click **Reset** and switch to **rule_based** mode if you want to continue without LLM
        - 🔑 Upgrade your Groq API plan for higher limits
        """)
        st.stop()

    # =================== 4 CONSOLIDATED TABS ===================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Dashboard",
        "🧠 AI Brain & Comms",
        "⚔️ AI vs Rules",
        "⚙️ Deep Dive & Logs",
    ])

    # =================== TAB 1: LIVE DASHBOARD ===================
    with tab1:
        if st.session_state.rec:
            rec = st.session_state.rec
            day_num  = rec.get('day', model.current_day)
            path     = rec.get('path', 'NORMAL')
            demand   = rec.get('demand', 0)
            fulfilled = rec.get('fulfilled', 0)
            inv_before = rec.get('inv_before', 0)
            inv_after  = rec.get('inv_after', 0)

            # ---- 1. Day header ----
            path_emoji = {"NORMAL": "🟢", "EMERGENCY": "🟡", "CRISIS": "🔴"}.get(path, "⚪")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:16px;padding:10px 0 6px 0">
                <span style="font-size:36px">📅</span>
                <h1 style="margin:0;font-size:2.2rem;font-weight:800;color:#f8fafc">
                    Day {day_num} &nbsp;— &nbsp;{path_emoji} &nbsp;{path} Path
                </h1>
            </div>""", unsafe_allow_html=True)

            # ---- 2. Status banner ----
            if fulfilled >= demand:
                _bg = "rgba(34,197,94,0.15)"; _bd = "#22c55e"; _tc = "#86efac"
                _msg = f"✅ All orders fulfilled! Shipped {fulfilled}/{demand} units. Stock: {inv_before} → {inv_after}"
            elif fulfilled > 0:
                _bg = "rgba(234,179,8,0.15)"; _bd = "#eab308"; _tc = "#fde68a"
                _msg = f"⚠️ Partial fulfillment. Shipped {fulfilled}/{demand} units. Stock: {inv_before} → {inv_after}"
            else:
                _bg = "rgba(239,68,68,0.15)"; _bd = "#ef4444"; _tc = "#fca5a5"
                _msg = f"❌ Stockout! Could not fulfill {demand} units. Stock: {inv_before} → {inv_after}"
            st.markdown(f"""
            <div style="background:{_bg};border:1.5px solid {_bd};border-radius:10px;padding:14px 20px;margin:10px 0 20px 0;font-size:14px;font-weight:600;color:{_tc}">
                {_msg}
            </div>""", unsafe_allow_html=True)

            # ---- 3. KPIs ----
            st.header("📊 Key Performance Indicators")
            mc = st.columns(6)
            kpi_items = [
                ('Fill Rate (%)', '📈'), ('Stock-out Rate (%)', '📉'),
                ('Avg Inventory', '📦'), ('Resilience Index', '🛡'),
                ('Customer Satisfaction', '⭐'), ('Avg Recovery Time (days)', '⏱')
            ]
            for col, (n, icon) in zip(mc, kpi_items):
                with col:
                    st.metric(f"{icon} {n}", f"{kpis.get(n, 0)}")

            # ---- 4. Inventory Over Time + Stock Health gauge ----
            if st.session_state.data:
                df = pd.DataFrame(st.session_state.data)

                c1, c2 = st.columns([3, 1])
                with c1:
                    st.subheader("📦 Inventory Over Time")
                    fig_inv = go.Figure()
                    fig_inv.add_trace(go.Scatter(
                        x=df['day'], y=df['inv_after'],
                        name="Inventory", fill='tozeroy',
                        line=dict(color='#3b82f6', width=2),
                        fillcolor='rgba(59,130,246,0.15)'))
                    fig_inv.add_hline(y=200, line_dash="dash", line_color="#eab308",
                        annotation_text="Reorder Point (200)")
                    fig_inv.update_layout(height=300, template='plotly_dark',
                        yaxis_title="Units", xaxis_title="Day",
                        margin=dict(t=20, b=40))
                    st.plotly_chart(fig_inv, use_container_width=True)

                with c2:
                    st.subheader("Health")
                    fig_g = render_gauge(model.warehouse.inventory)
                    st.plotly_chart(fig_g, use_container_width=True)
                    if model.warehouse.inventory <= 0:
                        st.error("🔴 EMPTY!")
                    elif model.warehouse.inventory < 200:
                        st.warning("🟡 LOW")
                    else:
                        st.success("🟢 HEALTHY")

                # ---- 5. Demand vs Fulfilled ----
                st.subheader("🛒 Demand vs Fulfilled")
                fig_d = go.Figure()
                fig_d.add_trace(go.Bar(x=df['day'], y=df['demand'],
                    name='Customers Wanted', marker_color='#ef4444', opacity=0.5))
                fig_d.add_trace(go.Bar(x=df['day'], y=df['fulfilled'],
                    name='We Shipped', marker_color='#22c55e', opacity=0.9))
                fig_d.update_layout(height=260, barmode='overlay', template='plotly_dark',
                    xaxis_title="Day", yaxis_title="Units",
                    legend=dict(orientation='v', x=1.01, y=0.9),
                    margin=dict(t=20, b=40))
                st.plotly_chart(fig_d, use_container_width=True)

                # ---- 6. Shipments In Transit ----
                st.subheader("📦 Shipments In Transit")
                render_shipments(model)

                # ---- 7. Workflow Path History ----
                st.subheader("🔀 Workflow Path History")
                _path_colors = {"NORMAL": "#22c55e", "EMERGENCY": "#eab308", "CRISIS": "#ef4444"}
                _path_nums   = {"NORMAL": 1, "EMERGENCY": 2, "CRISIS": 3}
                _path_labels = {1: "Normal", 2: "Emergency", 3: "Crisis"}
                _days  = [r['day'] for r in st.session_state.data]
                _paths = [r['path'] for r in st.session_state.data]
                _pnums = [_path_nums.get(p, 1) for p in _paths]
                _pclrs = [_path_colors.get(p, '#22c55e') for p in _paths]
                fig_wf = go.Figure()
                fig_wf.add_trace(go.Bar(
                    x=_days, y=_pnums,
                    marker_color=_pclrs,
                    text=_paths,
                    textposition='inside',
                    textfont=dict(color='white', size=11),
                    showlegend=False
                ))
                fig_wf.update_layout(
                    height=220, template='plotly_dark',
                    xaxis_title="Day",
                    yaxis=dict(
                        tickvals=[1, 2, 3],
                        ticktext=["Normal", "Emergency", "Crisis"],
                        range=[0, 3.5]
                    ),
                    margin=dict(t=10, b=40)
                )
                st.plotly_chart(fig_wf, use_container_width=True)

        else:
            # Welcome screen (no simulation run yet)
            st.markdown("""
            <div style="text-align:center;padding:60px 20px">
                <div style="font-size:64px;margin-bottom:16px">🏭</div>
                <h2 style="color:#f8fafc;margin-bottom:8px">AI Supply Chain Simulator</h2>
                <p style="color:#94a3b8;font-size:15px;max-width:500px;margin:0 auto 24px auto">
                    Powered by <strong>Walmart M5</strong> real sales data · <strong>Groq LLM</strong> · <strong>LSTM forecasting</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            _w1, _w2, _w3 = st.columns(3)
            _w1.info("**▶️ Click Next Day** in the sidebar to start the simulation")
            _w2.info("**🌪️ Add disruptions** like Hurricane or Road Block to stress-test")
            _w3.info("**🧠 Switch to AI Brain tab** to see agent reasoning and decisions")
    # =================== CHARTS & KPIs ===================
    # =================== WHY? (XAI) ===================
    with tab2:
        st.header("🧠 Explainable AI — Decision Transparency")
        st.markdown("Understand **why** every decision was made, **who** made it, and **how confident** the system is.")

        xai = model.xai
        sm = xai.get_summary()
        if sm['total_decisions'] == 0:
            st.info("▶️ Run the simulation first to see AI explanations.")
        else:
            # ---- Overview Metrics ----
            st.subheader("📊 Decision Overview")
            m1, m2, m3, m4, m5 = st.columns(5)
            llm_pct = sm['llm_decisions'] / max(sm['total_decisions'], 1) * 100
            rule_pct = 100 - llm_pct
            m1.metric("📋 Total Decisions", sm['total_decisions'])
            m2.metric("🧠 LLM Decisions", sm['llm_decisions'])
            m3.metric("📐 Rule Decisions", sm['total_decisions'] - sm['llm_decisions'])
            m4.metric("🎯 Avg Confidence", f"{sm['avg_confidence']:.0%}")
            m5.metric("📅 Days Simulated", model.current_day)

            # ---- AI vs Rules Split ----
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.subheader("🤖 AI vs Rules — Decision Ratio")
                fig_ratio = go.Figure(data=[go.Pie(
                    labels=['🧠 LLM (AI)', '📐 Rules'],
                    values=[sm['llm_decisions'], sm['total_decisions'] - sm['llm_decisions']],
                    marker_colors=['#8b5cf6', '#64748b'],
                    textinfo='label+percent+value',
                    hole=0.5,
                    textfont=dict(size=14)
                )])
                fig_ratio.update_layout(
                    height=300, template='plotly_dark',
                    title="Who Made the Decisions?",
                    annotations=[dict(text=f"{llm_pct:.0f}%<br>AI", x=0.5, y=0.5,
                                     font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig_ratio, use_container_width=True)

            with col_chart2:
                st.subheader("🎯 Confidence Distribution")
                # Collect confidence values per day
                all_confidences = []
                all_agents = []
                for day_i in range(1, model.current_day + 1):
                    day_decisions = xai.get_decision_chain(day_i)
                    for dd in day_decisions:
                        conf = dd.get('confidence', 0)
                        all_confidences.append(conf)
                        all_agents.append(dd.get('agent', '?'))

                if all_confidences:
                    fig_conf = go.Figure()
                    fig_conf.add_trace(go.Histogram(
                        x=all_confidences, nbinsx=10,
                        marker_color='#22c55e', opacity=0.8
                    ))
                    fig_conf.add_vline(x=sm['avg_confidence'], line_dash="dash",
                        line_color="#f59e0b",
                        annotation_text=f"Avg: {sm['avg_confidence']:.0%}")
                    fig_conf.update_layout(
                        height=300, template='plotly_dark',
                        xaxis_title="Confidence", yaxis_title="Count",
                        title="How Confident Were Decisions?"
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

            st.divider()

            # ---- Day Selector ----
            st.subheader("🔍 Explore Decisions by Day")
            if model.current_day > 1:
                sel = st.slider("Pick a day to inspect:", 1, model.current_day, model.current_day)
            else:
                sel = model.current_day

            decisions = xai.get_decision_chain(sel)
            if decisions:
                st.markdown(f"### 📅 Day {sel} — {len(decisions)} decisions made")

                for i, d in enumerate(decisions):
                    why = d.get('why', {})
                    path = d.get('workflow_path', '')
                    agent = d.get('agent', '?')
                    action = d.get('action', 'N/A')
                    dtype = d.get('decision_type', '')
                    conf = d.get('confidence', 0)
                    is_llm = d.get('is_llm_decision', False)

                    # Color coding
                    path_emoji = {"NORMAL": "🟢", "EMERGENCY": "🟡", "CRISIS": "🔴"}.get(path, "⚪")
                    agent_emoji = {"Demand": "📈", "Warehouse": "📦", "Supplier": "🏭", "Logistics": "🚚"}.get(agent, "🤖")
                    source_label = "🧠 LLM Decision" if is_llm else "📐 Rule-Based"
                    conf_color = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"

                    with st.expander(
                        f"{'━'*2} Step {i+1}: {agent_emoji} **{agent}** → {action[:60]} {path_emoji}",
                        expanded=(dtype not in ('forecast', 'delivery', 'fulfill_demand'))
                    ):
                        # Decision card with columns
                        dc1, dc2, dc3 = st.columns([2, 1, 1])
                        with dc1:
                            st.markdown(f"**🎯 Action:** {action}")
                            st.markdown(f"**📋 Type:** `{dtype}`")
                        with dc2:
                            st.markdown(f"**Source:** {source_label}")
                        with dc3:
                            st.markdown(f"**Confidence:** {conf_color} {conf:.0%}")

                        # WHY section
                        summary = why.get('summary', '')
                        reasoning = why.get('reasoning', '')
                        factors = why.get('contributing_factors', [])
                        alts = why.get('alternatives_considered', [])
                        triggered = why.get('triggered_by', '')
                        fallback = d.get('fallback_reason', '')

                        if summary:
                            st.info(f"💡 **Why:** {summary}")

                        if reasoning and reasoning != summary:
                            with st.expander("🧠 Full Reasoning"):
                                st.markdown(reasoning)

                        if factors:
                            st.markdown("**📊 Contributing Factors:**")
                            for fi, f in enumerate(factors):
                                st.markdown(f"  {fi+1}. {f}")

                        if alts:
                            st.markdown("**❌ Alternatives Rejected:**")
                            for a in alts:
                                opt = a.get('option', '') if isinstance(a, dict) else str(a)
                                reason = a.get('rejected_because', '') if isinstance(a, dict) else ''
                                st.markdown(f"  - ~~{opt}~~ — *{reason}*")

                        if triggered:
                            st.markdown(f"**⚡ Triggered by:** {triggered}")

                        if fallback:
                            st.warning(f"⚠️ Fallback: {fallback}")

            st.divider()

            # ---- Decision History Heatmap ----
            if model.current_day >= 3:
                st.subheader("📊 Decision Confidence Heatmap")
                st.markdown("Shows confidence across agents and days. Darker = more confident.")

                heatmap_data = []
                agents_seen = set()
                for day_i in range(1, model.current_day + 1):
                    day_decs = xai.get_decision_chain(day_i)
                    for dd in day_decs:
                        ag = dd.get('agent', '?')
                        agents_seen.add(ag)
                        heatmap_data.append({
                            'day': day_i,
                            'agent': ag,
                            'confidence': dd.get('confidence', 0),
                            'is_llm': dd.get('is_llm_decision', False)
                        })

                if heatmap_data:
                    agent_list = sorted(agents_seen)
                    days_range = list(range(1, model.current_day + 1))

                    # Build matrix
                    import numpy as np
                    z_data = []
                    for ag in agent_list:
                        row = []
                        for day_i in days_range:
                            matches = [h['confidence'] for h in heatmap_data
                                       if h['agent'] == ag and h['day'] == day_i]
                            row.append(np.mean(matches) if matches else 0)
                        z_data.append(row)

                    fig_heat = go.Figure(data=go.Heatmap(
                        z=z_data,
                        x=[f"Day {d}" for d in days_range],
                        y=agent_list,
                        colorscale='Viridis',
                        text=[[f"{v:.0%}" for v in row] for row in z_data],
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        colorbar=dict(title="Confidence")
                    ))
                    fig_heat.update_layout(
                        height=250, template='plotly_dark',
                        title="Agent Confidence Over Time",
                        xaxis_title="Day", yaxis_title="Agent"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

            # ---- Transparency Score ----
            st.divider()
            st.subheader("🏆 Transparency Score")
            explained = sum(1 for d in range(1, model.current_day + 1)
                           for dd in xai.get_decision_chain(d)
                           if dd.get('why', {}).get('summary'))
            total = sm['total_decisions']
            transparency = explained / max(total, 1) * 100

            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("📝 Decisions with Explanations", f"{explained}/{total}")
            tc2.metric("📊 Transparency Rate", f"{transparency:.0f}%")
            tc3.metric("🧠 AI Decision Rate", f"{llm_pct:.0f}%")

            if transparency >= 90:
                st.success(f"🏆 **Excellent transparency!** {transparency:.0f}% of decisions have clear explanations.")
            elif transparency >= 60:
                st.warning(f"⚠️ **Decent transparency:** {transparency:.0f}%. Some decisions lack detailed reasoning.")
            else:
                st.error(f"❌ **Low transparency:** {transparency:.0f}%. Many decisions need better explanations.")

    # =================== WORKFLOW ===================
    with tab2:
        st.divider()
        render_workflow(model, st.session_state.data)

        # ---- SECTION 3: AGENT GROUP CHAT (Internal Communications Log) ----
        st.divider()
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
            <span style="font-size:28px">💬</span>
            <div>
                <h2 style="margin:0;font-size:1.5rem">Internal Communications Log</h2>
                <p style="margin:0;color:#94a3b8;font-size:13px">Live inter-agent messages — every decision, broadcast, and order visible in real-time</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Agent avatar map (emoji used as avatar)
        _ag_av = {"Supplier": "🏭", "Warehouse": "📦", "Logistics": "🚚", "Demand": "📈", "System": "🌐", "Intel": "🧠"}
        _bus = model.bus if hasattr(model, 'bus') else None
        _msgs_raw = []
        if _bus and hasattr(_bus, 'history') and _bus.history:
            _msgs_raw = _bus.history
        elif _bus and hasattr(_bus, 'message_log') and _bus.message_log:
            _msgs_raw = _bus.message_log

        if _msgs_raw:
            _show_n_col, _order_col = st.columns([2, 1])
            _show_n = _show_n_col.selectbox("Messages to show", [10, 20, 50, 100], index=1, label_visibility="collapsed")
            _newest_first = _order_col.toggle("Newest first", value=False)
            _display_msgs = list(reversed(_msgs_raw[-_show_n:])) if _newest_first else _msgs_raw[-_show_n:]
            for _m in _display_msgs:
                # Support both object attributes and dict keys
                if isinstance(_m, dict):
                    _s = _m.get('sender', 'System')
                    _c = _m.get('content', str(_m))
                    _r = _m.get('recipient', 'all')
                    _day = _m.get('day', '')
                else:
                    _s = getattr(_m, 'sender', 'System')
                    _c = getattr(_m, 'content', str(_m))
                    _r = getattr(_m, 'recipient', 'all')
                    _day = getattr(_m, 'day', '')
                _avatar_emoji = _ag_av.get(_s, "🤖")
                _day_label = f" *(Day {_day})*" if _day else ""
                with st.chat_message(name=_s, avatar=_avatar_emoji):
                    st.markdown(f"**{_s}**{_day_label}")
                    st.markdown(f"@{_r} — {_c}")
        else:
            st.markdown("""
            <div style="background:#1e293b;border:1px dashed #334155;border-radius:12px;padding:32px;text-align:center;color:#64748b">
                <div style="font-size:40px;margin-bottom:8px">💬</div>
                <p style="margin:0;font-size:14px">No messages yet. Run the simulation to see live agent communications.</p>
            </div>
            """, unsafe_allow_html=True)

        # ---- SECTION 4: EXTERNAL INTELLIGENCE ----
        st.divider()
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
            <span style="font-size:28px">🛰️</span>
            <div>
                <h2 style="margin:0;font-size:1.5rem">External Intelligence</h2>
                <p style="margin:0;color:#94a3b8;font-size:13px">Live weather, news, and LLM risk analysis — informs agent decisions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        _ei1, _ei2 = st.columns([3, 1])
        with _ei2:
            _ref = st.button("🔄 Refresh Intelligence", use_container_width=True, key="t2_intel")
        if _ref or 'intel_cache' not in st.session_state:
            with st.spinner("Fetching live weather and news..."):
                try:
                    from news_search import get_intelligence_analyzer
                    _iao = get_intelligence_analyzer()
                    st.session_state['intel_cache'] = _iao.gather_intelligence(model.current_day)
                except Exception as _ex:
                    st.session_state['intel_cache'] = {}
                    st.warning(f"Intel fetch failed: {_ex}")

        _idat = st.session_state.get('intel_cache', {})
        _wt = _idat.get('weather', {})
        _lr = _idat.get('llm_risk_assessment', {})
        _news = _idat.get('news', [])

        if _wt:
            _sv = _wt.get('severity', 'normal')
            _sev_label = _sv.upper()
            _sc2 = '#22c55e' if _sv == 'normal' else '#eab308' if _sv == 'moderate' else '#ef4444'
            _sev_bg = 'rgba(34,197,94,0.15)' if _sv == 'normal' else 'rgba(234,179,8,0.15)' if _sv == 'moderate' else 'rgba(239,68,68,0.15)'
            _temp = _wt.get('temperature', 0)
            _fl = _wt.get('feels_like', _temp)
            _desc = _wt.get('description', '')
            _emoji = _wt.get('emoji', '☁️')
            _loc = _wt.get('location', 'Los Angeles, California').upper()
            _wind = _wt.get('wind_speed', 0)
            _gusts = _wt.get('wind_gusts', _wt.get('gusts', '—'))
            _rain = _wt.get('rain', 0)
            _hum = _wt.get('humidity', '—')

            st.markdown(f"""
            <div style="background:#1e293b;border-radius:14px;padding:24px;border:1px solid #334155;margin-bottom:16px">
                <p style="margin:0 0 12px 0;color:#94a3b8;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px">📍 {_loc}</p>
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px">
                    <div>
                        <div style="display:flex;align-items:center;gap:16px">
                            <span style="font-size:52px;line-height:1">{_emoji}</span>
                            <div>
                                <div style="font-size:44px;font-weight:800;color:#f8fafc;line-height:1">{_temp}°C</div>
                                <div style="font-size:13px;color:#94a3b8;margin-top:4px">Feels like {_fl}°C</div>
                                <div style="font-size:15px;color:#e2e8f0;margin-top:6px;font-weight:600">{_desc}</div>
                            </div>
                        </div>
                    </div>
                    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start">
                        <div style="text-align:center">
                            <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase">💨 Wind</div>
                            <div style="font-size:18px;font-weight:700;color:#f8fafc;margin-top:4px">{_wind} km/h</div>
                            <div style="font-size:11px;color:#64748b">Gusts {_gusts}</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase">💧 Rain</div>
                            <div style="font-size:18px;font-weight:700;color:#f8fafc;margin-top:4px">{_rain} mm</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase">💦 Humidity</div>
                            <div style="font-size:18px;font-weight:700;color:#f8fafc;margin-top:4px">{_hum}%</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase">⚠️ Severity</div>
                            <div style="margin-top:6px;background:{_sev_bg};color:{_sc2};border:1px solid {_sc2};border-radius:6px;padding:4px 14px;font-size:13px;font-weight:700">{_sev_label}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3-day forecast
            if _wt.get('forecast_3day'):
                _fcc = st.columns(3)
                for _fi, _fc in enumerate(_wt['forecast_3day'][:3]):
                    with _fcc[_fi]:
                        _fd = _fc.get('date', '')
                        _fdesc = _fc.get('description', '')
                        st.markdown(f"""
                        <div style="background:#1e293b;padding:16px;border-radius:10px;text-align:center;border:1px solid #334155;height:110px;display:flex;flex-direction:column;justify-content:center;gap:4px">
                            <div style="font-size:12px;color:#94a3b8;font-weight:600">{_fd}</div>
                            <div style="font-size:28px;margin:4px 0">{_fc.get('emoji','')}</div>
                            <div style="font-size:13px;color:#e2e8f0"><strong>{_fc.get('temp_max','')}° / {_fc.get('temp_min','')}°</strong></div>
                            <div style="font-size:11px;color:#64748b">{_fdesc}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1e293b;border:1px dashed #334155;border-radius:12px;padding:32px;text-align:center;color:#64748b;margin-bottom:16px">
                <div style="font-size:40px;margin-bottom:8px">🌤️</div>
                <p style="margin:0;font-size:14px">Click <strong>Refresh Intelligence</strong> to fetch live Los Angeles weather and supply chain news.</p>
            </div>
            """, unsafe_allow_html=True)

        # LLM Risk Analysis section
        if _lr:
            st.markdown("---")
            st.markdown("### 🧠 LLM Risk Analysis")
            _rl = _lr.get('risk_level', 'LOW')
            _rs = _lr.get('risk_score', 0)
            _rsrc = _lr.get('source', 'LLM')
            _dot = '🟢' if _rl == 'LOW' else '🟡' if _rl == 'MEDIUM' else '🔴'
            _rc1, _rc2, _rc3 = st.columns(3)
            _rc1.markdown(f"""
            <div style="padding:4px 0">
                <div style="font-size:12px;color:#94a3b8;margin-bottom:6px">Risk Level</div>
                <div style="font-size:26px;font-weight:800;color:#f8fafc">{_dot} {_rl}</div>
            </div>""", unsafe_allow_html=True)
            _rc2.markdown(f"""
            <div style="padding:4px 0">
                <div style="font-size:12px;color:#94a3b8;margin-bottom:6px">Risk Score</div>
                <div style="font-size:26px;font-weight:800;color:#f8fafc">{_rs}%</div>
            </div>""", unsafe_allow_html=True)
            _rc3.markdown(f"""
            <div style="padding:4px 0">
                <div style="font-size:12px;color:#94a3b8;margin-bottom:6px">Analysis Source</div>
                <div style="font-size:26px;font-weight:800;color:#f8fafc">🧠 {_rsrc}</div>
            </div>""", unsafe_allow_html=True)

            # Weather & News impact side by side
            _wi = _lr.get('weather_impact', '')
            _ni = _lr.get('news_impact', '')
            if _wi or _ni:
                _imp1, _imp2 = st.columns(2)
                with _imp1:
                    if _wi:
                        st.markdown(f"""
                        <div style="margin-top:12px">
                            <div style="font-size:13px;color:#f8fafc;margin-bottom:8px;font-weight:700">☀️ Weather Impact</div>
                            <div style="background:#1e3a5f;border-radius:8px;padding:16px;color:#bfdbfe;font-size:13px;line-height:1.7;border:1px solid #1e40af;min-height:80px">{_wi}</div>
                        </div>""", unsafe_allow_html=True)
                with _imp2:
                    if _ni:
                        st.markdown(f"""
                        <div style="margin-top:12px">
                            <div style="font-size:13px;color:#f8fafc;margin-bottom:8px;font-weight:700">🗞️ News Impact</div>
                            <div style="background:#1e3a5f;border-radius:8px;padding:16px;color:#bfdbfe;font-size:13px;line-height:1.7;border:1px solid #1e40af;min-height:80px">{_ni}</div>
                        </div>""", unsafe_allow_html=True)

            if _lr.get('recommendation'):
                st.markdown(f"""
                <div style="background:rgba(34,197,94,0.1);border:1px solid #22c55e;border-radius:10px;padding:16px 20px;margin-top:16px;font-size:13px;color:#86efac;line-height:1.7">
                    💡 <strong style="color:#4ade80">Recommendation:</strong> {_lr['recommendation']}
                </div>""", unsafe_allow_html=True)

        # Latest Supply Chain News
        if _news:
            st.markdown("---")
            st.markdown("### 📰 Latest Supply Chain News")
            for _article in _news[:5]:
                _title = _article.get('title', '')
                _source = _article.get('source', '')
                _url = _article.get('url', _article.get('href', '#'))
                if _title:
                    st.markdown(f"""
                    <div style="padding:10px 0;border-bottom:1px solid #1e293b;font-size:13px;color:#e2e8f0">
                        🗞️ {_title} — <em style="color:#94a3b8">{_source}</em>&nbsp;&nbsp;<a href="{_url}" target="_blank" style="color:#38bdf8;text-decoration:none;font-size:12px">Link ↗</a>
                    </div>""", unsafe_allow_html=True)

    # =================== TAB 3: AI vs RULES COMPARISON ===================
    with tab3:
        st.header("⚔️ AI Agents vs Simple Rules")
        st.markdown("""
        **Stress Test:** Both systems face the same demand & disruptions.  
        Lower starting stock + overlapping disruptions + demand spikes expose rule-based weaknesses.
        """)

        comp_days = st.slider("Days to simulate:", 10, 50, 30)
        if st.button("🏁 Run Comparison!", type="primary", use_container_width=True):
            try:
                import numpy as np
                with st.spinner("Running both simulations..."):
                    # Pre-generate FIXED demand sequence so both use identical demand
                    base_demands = []
                    temp_fc = st.session_state.fc
                    for _ in range(comp_days):
                        try:
                            pred = temp_fc.predict_next()
                            variance = pred * 0.2
                            d = max(1, int(pred + np.random.normal(0, variance)))
                        except Exception:
                            d = np.random.randint(50, 150)
                        base_demands.append(d)

                    # Add demand spikes during disruptions
                    demands = list(base_demands)
                    for i in range(comp_days):
                        # Spike during supplier disruption (days 5-10)
                        if 5 <= i <= 10:
                            demands[i] = int(demands[i] * 1.8)
                        # Spike during logistics disruption (days 15-20)
                        if 17 <= i <= 20:
                            demands[i] = int(demands[i] * 1.5)

                    # Create models with LOWER starting inventory (harder)
                    ai_m = init_model(st.session_state.dl, st.session_state.fc, 'agentic')
                    ai_m.warehouse.inventory = 300  # Lower start
                    ai_k = KPIEvaluator()
                    ai_d = []

                    rb_m = init_model(st.session_state.dl, st.session_state.fc, 'rule_based')
                    rb_m.warehouse.inventory = 300  # Same lower start
                    rb_k = KPIEvaluator()
                    rb_d = []

                    prog = st.progress(0)
                    disruption_log = []

                    for i in range(comp_days):
                        # Wave 1: Supplier disruption (day 5), during demand spike
                        if i == 4:
                            ai_m.inject_disruption('supplier', 5)
                            rb_m.inject_disruption('supplier', 5)
                            disruption_log.append((5, "🌪️ Supplier down"))

                        # Wave 2: Logistics disruption (day 15), overlapping recovery
                        if i == 14:
                            ai_m.inject_disruption('logistics', 5)
                            rb_m.inject_disruption('logistics', 5)
                            disruption_log.append((15, "🚧 Logistics down"))

                        # Wave 3: Double disruption (day 22)
                        if i == 21 and comp_days >= 25:
                            ai_m.inject_disruption('supplier', 3)
                            rb_m.inject_disruption('supplier', 3)
                            ai_m.inject_disruption('logistics', 3)
                            rb_m.inject_disruption('logistics', 3)
                            disruption_log.append((22, "💥 Both down"))

                        # Force IDENTICAL demand for both models
                        ai_m.daily_demand = demands[i]
                        rb_m.daily_demand = demands[i]

                        # Run AI step (demand already set, skip demand agent's random generation)
                        inv_before_ai = ai_m.warehouse.inventory
                        ai_m.step()
                        # Override with our fixed demand (step() may have changed it)
                        actual_ai_demand = demands[i]
                        ai_fulfilled = min(actual_ai_demand, inv_before_ai)
                        ai_stockout = ai_fulfilled < actual_ai_demand
                        ai_k.update(actual_ai_demand, ai_fulfilled, ai_m.warehouse.inventory,
                                    ai_stockout, day=ai_m.current_day,
                                    disruption_active=bool(ai_m.disruption_schedule))
                        ai_d.append({
                            'day': ai_m.current_day,
                            'inv_after': ai_m.warehouse.inventory,
                            'demand': actual_ai_demand,
                            'fulfilled': ai_fulfilled,
                            'stockout': ai_stockout,
                        })

                        # Run Rules step
                        inv_before_rb = rb_m.warehouse.inventory
                        rb_m.step()
                        actual_rb_demand = demands[i]
                        rb_fulfilled = min(actual_rb_demand, inv_before_rb)
                        rb_stockout = rb_fulfilled < actual_rb_demand
                        rb_k.update(actual_rb_demand, rb_fulfilled, rb_m.warehouse.inventory,
                                    rb_stockout, day=rb_m.current_day,
                                    disruption_active=bool(rb_m.disruption_schedule))
                        rb_d.append({
                            'day': rb_m.current_day,
                            'inv_after': rb_m.warehouse.inventory,
                            'demand': actual_rb_demand,
                            'fulfilled': rb_fulfilled,
                            'stockout': rb_stockout,
                        })

                        prog.progress((i + 1) / comp_days)
                    prog.empty()
            except GroqRateLimitError:
                st.error("""
                ## 🛑 GROQ RATE LIMIT EXCEEDED!
                
                The comparison could not complete because the Groq API rate limit was hit.
                
                **What to do:**
                - ⏳ Wait a few minutes for the rate limit to reset
                - 🔑 Upgrade your Groq API plan at https://console.groq.com/settings/billing
                """)
                st.stop()

            ak = ai_k.calculate_kpis()
            rk = rb_k.calculate_kpis()

            # Show scenario description
            st.markdown("### 📋 Scenario")
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Starting Inventory | **300** (low) |
            | Days | {comp_days} |
            | Disruptions | {len(disruption_log)} waves |
            | Demand Spikes | During disruptions (+50-80%) |
            """)
            for day, desc in disruption_log:
                st.markdown(f"- **Day {day}**: {desc}")

            # KPI comparison
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🤖 AI Agents")
                for n in ['Fill Rate (%)', 'Stock-out Rate (%)', 'Avg Inventory', 'Customer Satisfaction']:
                    st.metric(n, f"{ak.get(n, 0)}")
            with c2:
                st.subheader("📐 Simple Rules")
                for n in ['Fill Rate (%)', 'Stock-out Rate (%)', 'Avg Inventory', 'Customer Satisfaction']:
                    v = rk.get(n, 0)
                    d = v - ak.get(n, 0)
                    st.metric(n, f"{v}", delta=f"{d:+.1f} vs AI" if d != 0 else "Same")

            # Chart
            ai_df = pd.DataFrame(ai_d)
            rb_df = pd.DataFrame(rb_d)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ai_df['day'], y=ai_df['inv_after'],
                name='🤖 AI', line=dict(color='#8b5cf6', width=3)))
            fig.add_trace(go.Scatter(x=rb_df['day'], y=rb_df['inv_after'],
                name='📐 Rules', line=dict(color='#eab308', width=3, dash='dash')))
            # Add disruption markers
            for day, desc in disruption_log:
                fig.add_vline(x=day, line_dash="dot", line_color="#ef4444",
                    annotation_text=desc)
            fig.add_hline(y=0, line_color="#ef4444", line_width=2,
                annotation_text="STOCKOUT LINE")
            fig.update_layout(height=400, template='plotly_dark',
                xaxis_title="Day", yaxis_title="Inventory",
                title="Inventory Comparison Under Stress")
            st.plotly_chart(fig, use_container_width=True)

            # Stockout comparison chart
            st.subheader("📊 Stockout Events")
            ai_stockouts = sum(1 for r in ai_d if r['stockout'])
            rb_stockouts = sum(1 for r in rb_d if r['stockout'])
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=['AI Agents', 'Simple Rules'],
                y=[ai_stockouts, rb_stockouts],
                marker_color=['#8b5cf6', '#eab308'],
                text=[f'{ai_stockouts} days', f'{rb_stockouts} days'],
                textposition='auto'
            ))
            fig2.update_layout(height=250, template='plotly_dark',
                yaxis_title="Stockout Days", title="Days With Stockouts (lower = better)")
            st.plotly_chart(fig2, use_container_width=True)

            af = ak.get('Fill Rate (%)', 0)
            rf = rk.get('Fill Rate (%)', 0)
            if af > rf:
                st.success(f"🏆 **AI wins!** Fill rate {af}% vs {rf}%")
            elif rf > af:
                st.warning(f"📐 **Rules win this round!** {rf}% vs {af}%")
            else:
                st.info(f"🤝 **Tie!** Both at {af}%")

    # =================== TAB 4: DEEP DIVE & LOGS ===================
    with tab4:
        st.markdown("""
        <div style="padding:8px 0 20px 0">
            <h2 style="margin:0 0 6px 0;font-size:1.8rem">⚙️ Deep Dive &amp; Logs</h2>
            <p style="color:#94a3b8;font-size:13px;margin:0">Detailed views of data, architecture, memory, and forecast accuracy — organized in expandable sections</p>
        </div>
        """, unsafe_allow_html=True)

        # ---- Expander 1: Data Explorer ----
        with st.expander("🗂️ Data Explorer — Training Dataset", expanded=False):
            _dl = st.session_state.dl
            _hist = _dl.historical_data
            _summary = _dl.get_data_summary()
            if _hist is not None and _summary is not None:
                _sname = "Walmart M5 Forecasting Competition" if _summary['source']=='m5' else "Enhanced Synthetic Data"
                _semoji = "🏪" if _summary['source']=='m5' else "🔧"
                st.markdown(f"""
                ### {_semoji} Dataset: **{_sname}**
                | Property | Value |
                |----------|-------|
                | **Source** | {_sname} |
                | **Total Days** | {_summary['total_days']:,} |
                | **Date Range** | {_summary['date_range']} |
                | **Mean Daily Demand** | {_summary['mean_demand']} units |
                | **Std Deviation** | {_summary['std_demand']} units |
                | **Min / Max** | {_summary['min_demand']} / {_summary['max_demand']} units |
                """)
                if _summary['source']=='m5':
                    st.info("📋 **About M5:** Real daily Walmart sales from CA, TX, WI. Store CA\\_1, Item FOODS\\_3\\_090 — realistic demand with weekday/weekend cycles, seasonal trends, and promo spikes.")
                st.divider()
                _show = min(365, len(_hist))
                _rec = _hist.tail(_show)
                _fh = go.Figure()
                _fh.add_trace(go.Scatter(x=_rec['ds'], y=_rec['y'], name='Daily Demand', line=dict(color='#3b82f6', width=1), opacity=0.6))
                if len(_rec) >= 30:
                    _fh.add_trace(go.Scatter(x=_rec['ds'], y=_rec['y'].rolling(30).mean(), name='30-Day MA', line=dict(color='#f59e0b', width=3)))
                _fh.update_layout(height=320, template='plotly_dark', xaxis_title="Date", yaxis_title="Units Sold", title=f"Last {_show} Days of Demand Data")
                st.plotly_chart(_fh, use_container_width=True)
                _dc1, _dc2 = st.columns(2)
                with _dc1:
                    st.subheader("📊 Demand Distribution")
                    _fdh = go.Figure()
                    _fdh.add_trace(go.Histogram(x=_hist['y'], nbinsx=40, marker_color='#8b5cf6', opacity=0.8))
                    _fdh.add_vline(x=_hist['y'].mean(), line_dash="dash", line_color="#f59e0b", annotation_text=f"Mean: {_hist['y'].mean():.1f}")
                    _fdh.update_layout(height=280, template='plotly_dark', xaxis_title="Demand", yaxis_title="Frequency", showlegend=False)
                    st.plotly_chart(_fdh, use_container_width=True)
                with _dc2:
                    st.subheader("📅 Weekly Pattern")
                    _dn = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                    _wa = _hist.groupby('day_of_week')['y'].mean()
                    _fwk = go.Figure()
                    _fwk.add_trace(go.Bar(x=[_dn[i] for i in _wa.index], y=_wa.values, marker_color=['#3b82f6']*5+['#22c55e']*2, text=[f'{v:.0f}' for v in _wa.values], textposition='auto'))
                    _fwk.update_layout(height=280, template='plotly_dark', xaxis_title="Day", yaxis_title="Avg Demand", showlegend=False)
                    st.plotly_chart(_fwk, use_container_width=True)
                st.divider()
                st.subheader("🗂️ Raw Data (last 200 rows)")
                _ddf = _hist.tail(200).copy()
                _ddf['ds'] = _ddf['ds'].dt.strftime('%Y-%m-%d')
                _ddf.columns = ['Date','Demand','Day of Week','Month']
                _ddf['Day of Week'] = _ddf['Day of Week'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
                st.dataframe(_ddf, use_container_width=True, height=350)
            else:
                st.warning("No data loaded yet. Run a simulation step first.")

        # ---- Expander 2: System Architecture ----
        with st.expander("🧱 System Architecture", expanded=False):
            _ah = """
            <style>.ac{font-family:'Segoe UI',sans-serif;padding:20px;background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:16px;color:white}.ar{display:flex;justify-content:center;gap:20px;margin:15px 0;flex-wrap:wrap}.ab{padding:16px 24px;border-radius:12px;text-align:center;min-width:130px;transition:transform 0.2s}.ab:hover{transform:translateY(-4px);box-shadow:0 8px 25px rgba(0,0,0,0.4)}.ab h4{margin:0 0 4px 0;font-size:14px}.ab p{margin:0;font-size:11px;opacity:.8}.da{background:linear-gradient(135deg,#059669,#10b981)}.ml{background:linear-gradient(135deg,#7c3aed,#8b5cf6)}.ag{background:linear-gradient(135deg,#2563eb,#3b82f6)}.inf{background:linear-gradient(135deg,#d97706,#f59e0b)}.or{background:linear-gradient(135deg,#dc2626,#ef4444)}.ui{background:linear-gradient(135deg,#0891b2,#06b6d4)}.aw{text-align:center;font-size:22px;color:#64748b;margin:4px 0}.ll{color:#94a3b8;font-size:12px;text-transform:uppercase;letter-spacing:2px;margin:14px 0 4px;text-align:center}</style>
            <div class="ac"><div class="ll">🎯 Presentation</div><div class="ar"><div class="ab ui"><h4>🖥️ Streamlit</h4><p>4-tab UI</p></div><div class="ab ui"><h4>📊 Plotly Charts</h4><p>Visualization</p></div></div><div class="aw">⬆️⬇️</div><div class="ll">🔀 Orchestration</div><div class="ar"><div class="ab or"><h4>🔀 LangGraph</h4><p>NORMAL→EMERGENCY→CRISIS</p></div><div class="ab or"><h4>🧠 Explainability</h4><p>XAI decision records</p></div></div><div class="aw">⬆️⬇️</div><div class="ll">🤖 Agent Layer (Mesa)</div><div class="ar"><div class="ab ag"><h4>🏭 Supplier</h4><p>Order processing</p></div><div class="ab ag"><h4>📦 Warehouse</h4><p>Inventory mgmt</p></div><div class="ab ag"><h4>🚚 Logistics</h4><p>Shipment scheduling</p></div><div class="ab ag"><h4>📈 Demand</h4><p>Demand forecasting</p></div></div><div class="aw">⬆️⬇️</div><div class="ll">🧠 Intelligence</div><div class="ar"><div class="ab ml"><h4>🤖 Groq LLM</h4><p>LLaMA 3.1 8B</p></div><div class="ab ml"><h4>📊 LSTM</h4><p>TensorFlow/Keras</p></div><div class="ab inf"><h4>📡 Message Bus</h4><p>Inter-agent comms</p></div><div class="ab inf"><h4>🧠 ChromaDB</h4><p>Vector memory</p></div></div><div class="aw">⬆️⬇️</div><div class="ll">📂 Data Layer</div><div class="ar"><div class="ab da"><h4>🏪 M5 Walmart</h4><p>30,490 products × 1,941 days</p></div><div class="ab da"><h4>📝 Sim Logs</h4><p>Event history</p></div></div></div>"""
            import streamlit.components.v1 as components
            components.html(_ah, height=520, scrolling=True)
            st.subheader("🛠️ Technology Stack")
            _td = {'Component':['Agent Framework','LLM Provider','LLM Model','Orchestration','Memory/Vector DB','Forecasting','Dashboard','Visualization','Dataset','Language'],'Technology':['Mesa (ABM)','Groq Cloud','LLaMA 3.1 8B','LangGraph','ChromaDB','LSTM (TensorFlow)','Streamlit','Plotly','Walmart M5','Python 3.10'],'Purpose':['Multi-agent simulation','Fast LLM inference','Reasoning & decisions','Workflow state machine','Episodic agent memory','Time-series demand prediction','Interactive UI','Charts & graphs','Real retail data','Core language']}
            st.dataframe(pd.DataFrame(_td), use_container_width=True, hide_index=True)

        # ---- Expander 3: Agent Learning Memory ----
        with st.expander("🧠 Agent Learning Memory", expanded=False):
            st.markdown("Agents store past experiences in ChromaDB and recall them for better future decisions")
            if hasattr(model, 'get_memory_stats'):
                _ms = model.get_memory_stats()
                if _ms:
                    _mc = st.columns(len(_ms))
                    for _mi, (_an, _st) in enumerate(_ms.items()):
                        with _mc[_mi]:
                            st.metric(f"🤖 {_an}", f"{_st.get('total_episodes',0)} episodes")
                            st.progress(_st.get('success_rate',0), text=f"Success: {_st.get('success_rate',0):.0%}")
                    st.divider()
                    for _an, _st in _ms.items():
                        with st.expander(f"🧠 {_an} Memory", expanded=(_an=='Warehouse')):
                            st.markdown(f"Episodes: **{_st.get('total_episodes',0)}** | Success Rate: **{_st.get('success_rate',0):.1%}**")
                            _ao = None
                            for _nn, _oo in [('Supplier',model.supplier),('Warehouse',model.warehouse),('Logistics',model.logistics),('Demand',model.demand_agent)]:
                                if _nn==_an and hasattr(_oo,'memory'): _ao=_oo; break
                            if _ao and hasattr(_ao.memory,'episodes') and _ao.memory.episodes:
                                for _ep in _ao.memory.episodes[-3:]:
                                    _ok = '✅' if _ep.get('outcome',{}).get('success') else '❌'
                                    st.markdown(f"**Day {_ep.get('day','?')}** {_ok} — {_ep.get('decision','N/A')[:80]}")
                            else:
                                st.info("No episodes yet for this agent.")
                else:
                    st.info("No memory data yet. Run more simulation steps.")
            else:
                st.info("Memory not available. Switch to **agentic** mode.")

        # ---- Expander 4: ML Forecast Accuracy ----
        with st.expander("🎯 ML Forecast Accuracy", expanded=False):
            st.markdown("How well is the LSTM model predicting demand?")
            _fd = st.session_state.data
            if _fd and len(_fd) >= 3:
                import numpy as np
                _ac = [r.get('demand',0) for r in _fd]
                _fco = st.session_state.fc
                _pp = []
                for _pi, _pr in enumerate(_fd):
                    try: _pp.append(_fco.predict_next())
                    except: _pp.append(_ac[_pi] if _pi < len(_ac) else 100)
                _dl2 = [r.get('day',i+1) for i,r in enumerate(_fd)]
                _aa = np.array(_ac, dtype=float); _pa = np.array(_pp[:len(_ac)], dtype=float)
                _er = _aa - _pa
                _mae = np.mean(np.abs(_er)); _rmse = np.sqrt(np.mean(_er**2))
                _mpe = np.mean(np.abs(_er / np.maximum(_aa,1))) * 100
                _acc = max(0, 100 - _mpe)
                _fm1, _fm2, _fm3, _fm4 = st.columns(4)
                _fm1.metric("📏 MAE", f"{_mae:.1f} units", help="Mean Absolute Error")
                _fm2.metric("📐 RMSE", f"{_rmse:.1f} units")
                _fm3.metric("📊 MAPE", f"{_mpe:.1f}%")
                _fm4.metric("🎯 Accuracy", f"{_acc:.1f}%", delta='Good' if _acc > 85 else 'Needs improvement')
                _ffa = go.Figure()
                _ffa.add_trace(go.Scatter(x=_dl2, y=_ac, name='Actual Demand', line=dict(color='#3b82f6', width=3)))
                _ffa.add_trace(go.Scatter(x=_dl2, y=_pp[:len(_dl2)], name='Predicted', line=dict(color='#f59e0b', width=2, dash='dash')))
                _ffa.update_layout(height=320, template='plotly_dark', xaxis_title="Day", yaxis_title="Demand", title="Forecast vs Reality")
                st.plotly_chart(_ffa, use_container_width=True)
                _fec1, _fec2 = st.columns(2)
                with _fec1:
                    st.subheader("📊 Error Distribution")
                    _fef = go.Figure()
                    _fef.add_trace(go.Histogram(x=_er, nbinsx=20, marker_color='#8b5cf6', opacity=0.8))
                    _fef.add_vline(x=0, line_dash="dash", line_color="#22c55e", annotation_text="Perfect")
                    _fef.update_layout(height=260, template='plotly_dark', xaxis_title="Error", yaxis_title="Count")
                    st.plotly_chart(_fef, use_container_width=True)
                with _fec2:
                    st.subheader("📉 Cumulative Error")
                    _fef2 = go.Figure()
                    _fef2.add_trace(go.Scatter(x=_dl2, y=np.cumsum(np.abs(_er)), fill='tozeroy', line=dict(color='#ef4444', width=2)))
                    _fef2.update_layout(height=260, template='plotly_dark', xaxis_title="Day", yaxis_title="Cumulative |Error|")
                    st.plotly_chart(_fef2, use_container_width=True)
                if _acc >= 90: st.success(f"🏆 **Excellent!** {_acc:.1f}% — LSTM performing very well.")
                elif _acc >= 75: st.warning(f"⚠️ **Decent:** {_acc:.1f}% — captures trends but has variance.")
                else: st.error(f"❌ **Low:** {_acc:.1f}% — model may need more training data.")
            else:
                st.info("▶️ Run at least 3 simulation steps to see forecast accuracy.")


    # =================== SIDEBAR ===================
    with st.sidebar:
        st.header("🛑 HITL Settings")
        hitl_enabled = st.checkbox("Enable Approval Gate", value=True)
        hitl_threshold = st.slider("Approval Threshold (Units)", 100, 1000, 400, 50)
        
        # Apply config
        if hasattr(model, 'hitl_enabled'):
            model.hitl_enabled = hitl_enabled
            model.hitl_threshold = hitl_threshold

        st.header("⚙️ Settings")
        new_mode = st.selectbox("AI Mode", ['agentic', 'hybrid', 'rule_based'],
            index=['agentic', 'hybrid', 'rule_based'].index(st.session_state.mode))
        if new_mode != st.session_state.mode:
            st.session_state.mode = new_mode
            st.session_state.model = init_model(st.session_state.dl, st.session_state.fc, new_mode)
            st.session_state.kpi = KPIEvaluator()
            st.session_state.data = []
            st.session_state.rec = None
            st.rerun()

        st.divider()
        st.markdown("""
        ### 📚 Glossary
        - 📦 **Inventory** = Stock in warehouse
        - 📈 **Fill Rate** = % of orders we could ship
        - 🔄 **Reorder Point** = 200 units (time to reorder)
        - 🏆 **Score** = Overall supply chain health
        - 🟢 **Normal** = Everything OK
        - 🟡 **Emergency** = Stock getting low
        - 🔴 **Crisis** = Stockout or disruption!
        """)

        # Active disruptions
        if model.disruption_schedule:
            st.markdown("### ⚠️ Active Disruptions")
            for at, ed in model.disruption_schedule.items():
                rem = ed - model.current_day
                if rem > 0:
                    st.error(f"**{at.title()}**: {rem} days left")
        else:
            st.success("✅ All systems operational")


if __name__ == "__main__":
    try:
        main()
    except GroqRateLimitError:
        st.error("""
        ## 🛑 GROQ RATE LIMIT EXCEEDED!
        
        **The simulation was stopped** because the Groq API rate limit was hit.
        
        The system does NOT fall back to rule-based mode — it strictly uses the LLM as requested.
        
        **What to do:**
        - ⏳ Wait a few minutes for the rate limit to reset
        - 🔄 Click **Reset** in sidebar and switch to **rule_based** mode to continue without LLM
        - 🔑 Upgrade your Groq API plan at https://console.groq.com/settings/billing
        """)
