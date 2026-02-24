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
    st.title("🤖 Agentic AI Supply Chain")

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

    # =================== TABS ===================
    tab1, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "📖 Day Briefing",
        "📊 Charts & KPIs",
        "🧠 Why? (XAI)",
        "🔀 Workflow",
        "⚔️ AI vs Rules",
        "📂 Data Explorer",
        "💰 Cost Analysis",
        "🏗️ Architecture",
        "📡 Agent Comms",
        "🧠 Memory",
        "🎯 Forecast",
    ])

    # =================== TAB 1: DAY BRIEFING (all-in-one) ===================
    with tab1:
        if st.session_state.rec:
            # 1. Supply chain map — visual overview
            render_map(model)

            # 2. Agent status cards — quick status
            st.markdown("---")
            render_agent_cards(model, st.session_state.rec)

            # 3. Shipment tracker
            st.subheader("📦 Shipments In Transit")
            render_shipments(model)

            st.markdown("---")

            # 4. Full day narrative — data source, scenario, decisions
            render_narrative(st.session_state.rec, model)

            # 5. Prediction for next day
            if model.warehouse.inventory > 0:
                avg_dem = model.daily_demand if model.daily_demand > 0 else 100
                days_left = model.warehouse.inventory / max(avg_dem, 1)
                will_ok = model.warehouse.inventory > avg_dem
                st.markdown("---")
                st.markdown("### 🔮 What might happen tomorrow?")
                st.markdown(f"- Expected demand: ~{avg_dem} units")
                st.markdown(f"- Current stock: {model.warehouse.inventory} units")
                st.markdown(f"- Days of stock left: **{days_left:.1f}**")
                if will_ok:
                    st.markdown(f"- Will we have enough? ✅ **Yes** ({model.warehouse.inventory - avg_dem} to spare)")
                else:
                    st.markdown(f"- Will we have enough? ❌ **No — stockout likely!**")

            # 6. Timeline
            if len(st.session_state.data) > 1:
                st.markdown("---")
                st.markdown("### 📜 Recent Timeline")
                for r in reversed(st.session_state.data[-8:]):
                    pe = {"NORMAL": "🟢", "EMERGENCY": "🟡", "CRISIS": "🔴"}.get(r['path'], "⚪")
                    so_icon = "❌" if r['stockout'] else "✅"
                    src_date = r.get('source_date', '')
                    date_str = f" ({src_date})" if src_date else ""
                    st.markdown(
                        f"{pe} **Day {r['day']}**{date_str} {so_icon} — "
                        f"Demand: {r['demand']} | Stock: {r['inv_after']} | {r['path']}")
        else:
            st.markdown("""
            ### 👋 Welcome to the AI Supply Chain Simulator!
            
            **What is this?** A simulation of a real supply chain powered by **Walmart M5 sales data**.
            AI agents use an **LLM (Groq)** + **LSTM forecasting** to make decisions about 
            ordering, shipping, and inventory management.
            
            **How to use:**
            1. Click **▶️ Next Day** to advance one day
            2. The **Day Briefing** shows you exactly:
               - 📂 Which dataset date is being used
               - 🎯 The full scenario (inventory, demand, supply status)
               - 🤖 Step-by-step agent decisions 
               - 💭 Raw LLM reasoning
            
            **Try disruptions** to stress-test the AI:
            - 🌪️ **Hurricane** — Supplier can't send goods for 4 days
            - 🚧 **Road Block** — Trucks can't deliver for 3 days  
            - 📈 **Demand Spike** — Suddenly customers want more!
            """)

    # =================== TAB 3: CHARTS ===================
    with tab3:
        if st.session_state.data:
            # KPIs
            st.header("📊 Key Performance Indicators")
            mc = st.columns(6)
            items = [
                ('Fill Rate (%)', '📈'), ('Stock-out Rate (%)', '📉'),
                ('Avg Inventory', '📦'), ('Resilience Index', '🛡'),
                ('Customer Satisfaction', '⭐'), ('Avg Recovery Time (days)', '⏱')
            ]
            for col, (n, icon) in zip(mc, items):
                with col:
                    st.metric(f"{icon} {n}", f"{kpis.get(n, 0)}")

            df = pd.DataFrame(st.session_state.data)

            # Inventory + Gauge
            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader("📦 Inventory Over Time")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['day'], y=df['inv_after'],
                    name="Inventory", fill='tozeroy',
                    line=dict(color='#3b82f6', width=2),
                    fillcolor='rgba(59,130,246,0.15)'))
                fig.add_hline(y=200, line_dash="dash", line_color="#eab308",
                    annotation_text="Reorder Point (200)")
                fig.update_layout(height=300, template='plotly_dark',
                    yaxis_title="Units", xaxis_title="Day")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("Stock Health")
                fig = render_gauge(model.warehouse.inventory)
                st.plotly_chart(fig, use_container_width=True)
                if model.warehouse.inventory <= 0:
                    st.error("🔴 EMPTY!")
                elif model.warehouse.inventory < 200:
                    st.warning("🟡 LOW")
                else:
                    st.success("🟢 HEALTHY")

            # Demand vs Fulfilled
            st.subheader("🛒 Demand vs Fulfilled")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['day'], y=df['demand'],
                name='Customers Wanted', marker_color='#ef4444', opacity=0.5))
            fig.add_trace(go.Bar(x=df['day'], y=df['fulfilled'],
                name='We Shipped', marker_color='#22c55e', opacity=0.9))
            fig.update_layout(height=250, barmode='overlay', template='plotly_dark',
                xaxis_title="Day", yaxis_title="Units")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the simulation to see charts.")

    # =================== TAB 4: XAI ===================
    with tab4:
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

    # =================== TAB 5: WORKFLOW ===================
    with tab5:
        render_workflow(model, st.session_state.data)

    # =================== TAB 6: COMPARISON ===================
    with tab6:
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

    # =================== TAB 7: DATA EXPLORER ===================
    with tab7:
        st.header("📂 Data Explorer — Training Dataset")

        dl = st.session_state.dl
        hist = dl.historical_data
        summary = dl.get_data_summary()

        if hist is not None and summary is not None:
            # Dataset info card
            source_name = "Walmart M5 Forecasting Competition" if summary['source'] == 'm5' else "Enhanced Synthetic Data"
            source_emoji = "🏪" if summary['source'] == 'm5' else "🔧"

            st.markdown(f"""
            ### {source_emoji} Dataset: **{source_name}**
            
            | Property | Value |
            |----------|-------|
            | **Source** | {source_name} |
            | **Total Days** | {summary['total_days']:,} |
            | **Date Range** | {summary['date_range']} |
            | **Mean Daily Demand** | {summary['mean_demand']} units |
            | **Std Deviation** | {summary['std_demand']} units |
            | **Min Demand** | {summary['min_demand']} units |
            | **Max Demand** | {summary['max_demand']} units |
            """)

            if summary['source'] == 'm5':
                st.info("""
                📋 **About M5 Dataset:** The [Walmart M5 Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) 
                contains **real daily sales** data from Walmart stores across California, Texas, and Wisconsin.
                We use **Store CA_1, Item FOODS_3_090** — a food product with realistic demand patterns
                including weekday/weekend patterns, seasonal trends, and promotional spikes.
                """)

            st.divider()

            # Historical demand chart
            st.subheader("📈 Historical Demand Over Time")
            fig = go.Figure()

            # Show last 365 days for clarity (or all if less)
            show_days = min(365, len(hist))
            recent = hist.tail(show_days)

            fig.add_trace(go.Scatter(
                x=recent['ds'], y=recent['y'],
                name='Daily Demand',
                line=dict(color='#3b82f6', width=1),
                opacity=0.6
            ))

            # 30-day moving average
            if len(recent) >= 30:
                ma = recent['y'].rolling(30).mean()
                fig.add_trace(go.Scatter(
                    x=recent['ds'], y=ma,
                    name='30-Day Moving Average',
                    line=dict(color='#f59e0b', width=3)
                ))

            fig.update_layout(
                height=350, template='plotly_dark',
                xaxis_title="Date", yaxis_title="Units Sold",
                title=f"Last {show_days} Days of Demand Data"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Two columns: distribution + patterns
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("📊 Demand Distribution")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=hist['y'], nbinsx=40,
                    marker_color='#8b5cf6', opacity=0.8,
                    name='Frequency'
                ))
                fig_hist.add_vline(x=hist['y'].mean(), line_dash="dash",
                    line_color="#f59e0b",
                    annotation_text=f"Mean: {hist['y'].mean():.1f}")
                fig_hist.update_layout(
                    height=300, template='plotly_dark',
                    xaxis_title="Daily Demand (units)",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with c2:
                st.subheader("📅 Weekly Pattern")
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                weekly_avg = hist.groupby('day_of_week')['y'].mean()
                fig_week = go.Figure()
                fig_week.add_trace(go.Bar(
                    x=[day_names[i] for i in weekly_avg.index],
                    y=weekly_avg.values,
                    marker_color=['#3b82f6', '#3b82f6', '#3b82f6', '#3b82f6',
                                  '#3b82f6', '#22c55e', '#22c55e'],
                    text=[f'{v:.0f}' for v in weekly_avg.values],
                    textposition='auto'
                ))
                fig_week.update_layout(
                    height=300, template='plotly_dark',
                    xaxis_title="Day of Week",
                    yaxis_title="Avg Demand",
                    showlegend=False
                )
                st.plotly_chart(fig_week, use_container_width=True)

            # Monthly pattern
            st.subheader("📆 Monthly Demand Pattern")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_avg = hist.groupby('month')['y'].mean()
            fig_month = go.Figure()
            fig_month.add_trace(go.Bar(
                x=[month_names[i - 1] for i in monthly_avg.index],
                y=monthly_avg.values,
                marker=dict(
                    color=monthly_avg.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Demand")
                ),
                text=[f'{v:.0f}' for v in monthly_avg.values],
                textposition='auto'
            ))
            fig_month.update_layout(
                height=300, template='plotly_dark',
                xaxis_title="Month",
                yaxis_title="Avg Demand"
            )
            st.plotly_chart(fig_month, use_container_width=True)

            st.divider()

            # Raw data table
            st.subheader("🗂️ Raw Data (scrollable)")
            st.markdown(f"Showing last **200 rows** of {len(hist):,} total rows")
            display_df = hist.tail(200).copy()
            display_df['ds'] = display_df['ds'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Date', 'Demand', 'Day of Week', 'Month']
            day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            display_df['Day of Week'] = display_df['Day of Week'].map(day_map)
            st.dataframe(display_df, use_container_width=True, height=400)

        else:
            st.warning("No data loaded yet. The data will be available after the first simulation step.")

    # =================== TAB 8: COST ANALYSIS ===================
    with tab8:
        st.header("💰 Cost Analysis — Financial Impact")
        st.markdown("Real-time supply chain costs based on simulation data")

        # Cost parameters
        HOLDING_COST_PER_UNIT = 0.50   # $/unit/day
        STOCKOUT_COST_PER_UNIT = 15.0  # $/unit lost sale
        SHIPPING_COST_PER_UNIT = 2.0   # $/unit shipped
        ORDER_FIXED_COST = 50.0        # $/order placed

        data = st.session_state.data
        if data and len(data) > 0:
            df = pd.DataFrame(data)
            days_run = len(df)

            # Calculate costs
            total_holding = sum(r.get('inv_after', 0) * HOLDING_COST_PER_UNIT for r in data)
            total_stockout = sum(
                max(0, r.get('demand', 0) - r.get('fulfilled', 0)) * STOCKOUT_COST_PER_UNIT
                for r in data
            )
            total_shipping = sum(r.get('fulfilled', 0) * SHIPPING_COST_PER_UNIT for r in data)
            num_orders = sum(1 for r in data if r.get('fulfilled', 0) > 0)
            total_order_cost = num_orders * ORDER_FIXED_COST
            total_cost = total_holding + total_stockout + total_shipping + total_order_cost

            # Cost parameters display
            with st.expander("📋 Cost Parameters (per unit)", expanded=False):
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Holding Cost", f"${HOLDING_COST_PER_UNIT}/unit/day")
                p2.metric("Stockout Penalty", f"${STOCKOUT_COST_PER_UNIT}/unit")
                p3.metric("Shipping Cost", f"${SHIPPING_COST_PER_UNIT}/unit")
                p4.metric("Order Fixed Cost", f"${ORDER_FIXED_COST}/order")

            # Total cost KPIs
            st.subheader("📊 Cost Summary")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("💵 Total Cost", f"${total_cost:,.0f}")
            k2.metric("📦 Holding", f"${total_holding:,.0f}")
            k3.metric("❌ Stockout", f"${total_stockout:,.0f}")
            k4.metric("🚚 Shipping", f"${total_shipping:,.0f}")
            k5.metric("📝 Orders", f"${total_order_cost:,.0f}")

            # Cost breakdown pie chart
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🥧 Cost Breakdown")
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Holding', 'Stockout Penalty', 'Shipping', 'Order Costs'],
                    values=[total_holding, total_stockout, total_shipping, total_order_cost],
                    marker_colors=['#3b82f6', '#ef4444', '#22c55e', '#f59e0b'],
                    textinfo='label+percent',
                    hole=0.4
                )])
                fig_pie.update_layout(height=350, template='plotly_dark',
                    title="Where Does Money Go?")
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                st.subheader("📈 Daily Cost Trend")
                daily_holding = [r.get('inv_after', 0) * HOLDING_COST_PER_UNIT for r in data]
                daily_stockout = [max(0, r.get('demand', 0) - r.get('fulfilled', 0)) * STOCKOUT_COST_PER_UNIT for r in data]
                daily_shipping = [r.get('fulfilled', 0) * SHIPPING_COST_PER_UNIT for r in data]
                days_list = [r.get('day', i+1) for i, r in enumerate(data)]

                fig_cost = go.Figure()
                fig_cost.add_trace(go.Scatter(x=days_list, y=daily_holding,
                    name='Holding', fill='tozeroy',
                    line=dict(color='#3b82f6'), stackgroup='costs'))
                fig_cost.add_trace(go.Scatter(x=days_list, y=daily_stockout,
                    name='Stockout', fill='tonexty',
                    line=dict(color='#ef4444'), stackgroup='costs'))
                fig_cost.add_trace(go.Scatter(x=days_list, y=daily_shipping,
                    name='Shipping', fill='tonexty',
                    line=dict(color='#22c55e'), stackgroup='costs'))
                fig_cost.update_layout(height=350, template='plotly_dark',
                    xaxis_title="Day", yaxis_title="Cost ($)",
                    title="Stacked Daily Costs")
                st.plotly_chart(fig_cost, use_container_width=True)

            # Cost per unit fulfilled
            total_fulfilled = sum(r.get('fulfilled', 0) for r in data)
            if total_fulfilled > 0:
                cost_per_unit = total_cost / total_fulfilled
                st.metric("💲 Cost Per Unit Fulfilled", f"${cost_per_unit:.2f}",
                          help="Total cost divided by total units successfully shipped to customers")
        else:
            st.info("▶️ Run the simulation to see cost analysis!")

    # =================== TAB 9: ARCHITECTURE ===================
    with tab9:
        st.header("🏗️ System Architecture")
        st.markdown("Interactive view of how all components connect")

        arch_html = """
        <style>
            .arch-container { 
                font-family: 'Segoe UI', sans-serif; 
                padding: 20px; 
                background: linear-gradient(135deg, #0f172a, #1e293b); 
                border-radius: 16px;
                color: white;
            }
            .arch-row { display: flex; justify-content: center; gap: 20px; margin: 15px 0; flex-wrap: wrap; }
            .arch-box {
                padding: 16px 24px; border-radius: 12px; text-align: center;
                min-width: 140px; transition: transform 0.2s, box-shadow 0.2s;
                cursor: pointer; position: relative;
            }
            .arch-box:hover { transform: translateY(-4px); box-shadow: 0 8px 25px rgba(0,0,0,0.4); }
            .arch-box h4 { margin: 0 0 4px 0; font-size: 14px; }
            .arch-box p { margin: 0; font-size: 11px; opacity: 0.8; }
            .data { background: linear-gradient(135deg, #059669, #10b981); }
            .ml { background: linear-gradient(135deg, #7c3aed, #8b5cf6); }
            .agent { background: linear-gradient(135deg, #2563eb, #3b82f6); }
            .infra { background: linear-gradient(135deg, #d97706, #f59e0b); }
            .orch { background: linear-gradient(135deg, #dc2626, #ef4444); }
            .ui { background: linear-gradient(135deg, #0891b2, #06b6d4); }
            .arrow { text-align: center; font-size: 24px; color: #64748b; margin: 5px 0; }
            .layer-label { 
                color: #94a3b8; font-size: 12px; text-transform: uppercase; 
                letter-spacing: 2px; margin: 15px 0 5px; text-align: center; 
            }
        </style>
        <div class="arch-container">
            <div class="layer-label">🎯 Presentation Layer</div>
            <div class="arch-row">
                <div class="arch-box ui">
                    <h4>🖥️ Streamlit Dashboard</h4>
                    <p>12 interactive tabs</p>
                </div>
                <div class="arch-box ui">
                    <h4>📊 Plotly Charts</h4>
                    <p>Real-time visualization</p>
                </div>
            </div>
            <div class="arrow">⬆️ ⬇️</div>
            
            <div class="layer-label">🔀 Orchestration Layer</div>
            <div class="arch-row">
                <div class="arch-box orch">
                    <h4>🔀 LangGraph</h4>
                    <p>State machine workflow</p>
                    <p>NORMAL → EMERGENCY → CRISIS</p>
                </div>
                <div class="arch-box orch">
                    <h4>🧠 Explainability</h4>
                    <p>XAI decision records</p>
                    <p>WHY explanations</p>
                </div>
            </div>
            <div class="arrow">⬆️ ⬇️</div>

            <div class="layer-label">🤖 Agent Layer (Mesa Framework)</div>
            <div class="arch-row">
                <div class="arch-box agent">
                    <h4>🏭 Supplier Agent</h4>
                    <p>Order processing</p>
                </div>
                <div class="arch-box agent">
                    <h4>📦 Warehouse Agent</h4>
                    <p>Inventory management</p>
                </div>
                <div class="arch-box agent">
                    <h4>🚚 Logistics Agent</h4>
                    <p>Shipment scheduling</p>
                </div>
                <div class="arch-box agent">
                    <h4>📈 Demand Agent</h4>
                    <p>Demand forecasting</p>
                </div>
            </div>
            <div class="arrow">⬆️ ⬇️</div>

            <div class="layer-label">🧠 Intelligence Layer</div>
            <div class="arch-row">
                <div class="arch-box ml">
                    <h4>🤖 Groq LLM</h4>
                    <p>LLaMA 3.1 8B Instant</p>
                    <p>via LangChain</p>
                </div>
                <div class="arch-box ml">
                    <h4>📊 LSTM Forecaster</h4>
                    <p>TensorFlow/Keras</p>
                    <p>Demand prediction</p>
                </div>
                <div class="arch-box infra">
                    <h4>📡 Message Bus</h4>
                    <p>Inter-agent comms</p>
                </div>
                <div class="arch-box infra">
                    <h4>🧠 ChromaDB</h4>
                    <p>Vector memory</p>
                    <p>Episode storage</p>
                </div>
            </div>
            <div class="arrow">⬆️ ⬇️</div>

            <div class="layer-label">📂 Data Layer</div>
            <div class="arch-row">
                <div class="arch-box data">
                    <h4>🏪 M5 Walmart Dataset</h4>
                    <p>30,490 products × 1,941 days</p>
                </div>
                <div class="arch-box data">
                    <h4>📝 Simulation Logs</h4>
                    <p>Event history & metrics</p>
                </div>
            </div>
        </div>
        """
        import streamlit.components.v1 as components
        components.html(arch_html, height=700, scrolling=True)

        # Tech stack summary
        st.subheader("🛠️ Technology Stack")
        tech_data = {
            'Component': ['Agent Framework', 'LLM Provider', 'LLM Model', 'Orchestration',
                        'Memory/Vector DB', 'Forecasting', 'Dashboard', 'Visualization',
                        'Dataset', 'Language'],
            'Technology': ['Mesa (ABM)', 'Groq Cloud', 'LLaMA 3.1 8B', 'LangGraph',
                         'ChromaDB', 'LSTM (TensorFlow)', 'Streamlit', 'Plotly',
                         'Walmart M5', 'Python 3.10'],
            'Purpose': ['Multi-agent simulation', 'Fast LLM inference', 'Reasoning & decisions',
                       'Workflow state machine', 'Episodic agent memory',
                       'Time-series demand prediction', 'Interactive UI',
                       'Charts & graphs', 'Real retail data', 'Core language']
        }
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    # =================== TAB 10: AGENT COMMUNICATIONS ===================
    with tab10:
        st.header("📡 Agent Communications — Message Bus")
        st.markdown("Real-time view of inter-agent messages")

        bus = model.bus if hasattr(model, 'bus') else None
        if bus:
            stats = bus.get_stats()
            c1, c2, c3 = st.columns(3)
            c1.metric("📨 Total Messages", stats.get('total_messages', 0))
            c2.metric("📬 Active Inboxes", len(stats.get('inbox_sizes', {})))
            c3.metric("📋 Message Types", len(stats.get('message_types', [])))

            # Message type breakdown
            if stats.get('message_types'):
                st.subheader("📊 Message Types")
                types = stats['message_types']
                fig_types = go.Figure(data=[go.Bar(
                    x=list(types.keys()) if isinstance(types, dict) else types,
                    y=list(types.values()) if isinstance(types, dict) else [1]*len(types),
                    marker_color='#8b5cf6'
                )])
                fig_types.update_layout(height=250, template='plotly_dark',
                    xaxis_title="Message Type", yaxis_title="Count")
                st.plotly_chart(fig_types, use_container_width=True)

            # Recent messages
            st.subheader("💬 Recent Messages")
            all_messages = []
            if hasattr(bus, 'history'):
                for msg in bus.history[-30:]:
                    all_messages.append({
                        'From': msg.sender if hasattr(msg, 'sender') else '?',
                        'To': msg.recipient if hasattr(msg, 'recipient') else '?',
                        'Type': msg.msg_type if hasattr(msg, 'msg_type') else '?',
                        'Content': str(msg.content)[:80] if hasattr(msg, 'content') else '?',
                    })
            if all_messages:
                st.dataframe(pd.DataFrame(all_messages), use_container_width=True, hide_index=True)
            else:
                # Try to show inbox contents
                for agent_name in ['Supplier', 'Warehouse', 'Logistics', 'Demand']:
                    msgs = bus.get_messages(agent_name)
                    for msg in msgs[-5:]:
                        all_messages.append({
                            'To': agent_name,
                            'From': msg.sender if hasattr(msg, 'sender') else '?',
                            'Type': msg.msg_type if hasattr(msg, 'msg_type') else '?',
                            'Content': str(msg.content)[:80] if hasattr(msg, 'content') else '?',
                        })
                if all_messages:
                    st.dataframe(pd.DataFrame(all_messages), use_container_width=True, hide_index=True)
                else:
                    st.info("No messages yet. Run a few simulation steps to see agent communication.")

            # Agent communication flow diagram
            st.subheader("🔄 Communication Flow")
            flow_html = """
            <div style="text-align:center; padding:20px; background:#1e293b; border-radius:12px; color:white; font-family:sans-serif;">
                <div style="display:flex; justify-content:center; align-items:center; gap:10px; flex-wrap:wrap;">
                    <div style="padding:12px 20px; background:#2563eb; border-radius:8px;">📈 Demand</div>
                    <div style="font-size:20px;">→ forecast →</div>
                    <div style="padding:12px 20px; background:#059669; border-radius:8px;">📦 Warehouse</div>
                    <div style="font-size:20px;">→ order →</div>
                    <div style="padding:12px 20px; background:#d97706; border-radius:8px;">🏭 Supplier</div>
                    <div style="font-size:20px;">→ ship →</div>
                    <div style="padding:12px 20px; background:#7c3aed; border-radius:8px;">🚚 Logistics</div>
                    <div style="font-size:20px;">→ deliver →</div>
                    <div style="padding:12px 20px; background:#059669; border-radius:8px;">📦 Warehouse</div>
                </div>
                <p style="margin-top:10px; color:#94a3b8;">Each arrow is a message on the bus ↑</p>
            </div>
            """
            components.html(flow_html, height=120)
        else:
            st.info("Message bus not available in this mode.")

    # =================== TAB 11: MEMORY EXPLORER ===================
    with tab11:
        st.header("🧠 Memory Explorer — What Agents Have Learned")
        st.markdown("Agents store past experiences in ChromaDB and recall them for better decisions")

        if hasattr(model, 'get_memory_stats'):
            mem_stats = model.get_memory_stats()

            if mem_stats:
                # Overview metrics
                cols = st.columns(len(mem_stats))
                for i, (agent_name, stats) in enumerate(mem_stats.items()):
                    with cols[i]:
                        total = stats.get('total_episodes', 0)
                        rate = stats.get('success_rate', 0)
                        st.metric(f"🤖 {agent_name}", f"{total} episodes")
                        st.progress(rate, text=f"Success: {rate:.0%}")

                st.divider()

                # Per-agent details
                for agent_name, stats in mem_stats.items():
                    with st.expander(f"🧠 {agent_name} Agent Memory", expanded=(agent_name == 'Warehouse')):
                        st.markdown(f"""
                        | Metric | Value |
                        |--------|-------|
                        | Total Episodes | **{stats.get('total_episodes', 0)}** |
                        | Success Rate | **{stats.get('success_rate', 0):.1%}** |
                        """)

                        # Try to show actual memory episodes
                        agent_obj = None
                        for name, obj in [('Supplier', model.supplier), ('Warehouse', model.warehouse),
                                        ('Logistics', model.logistics), ('Demand', model.demand_agent)]:
                            if name == agent_name and hasattr(obj, 'memory'):
                                agent_obj = obj
                                break

                        if agent_obj and hasattr(agent_obj.memory, 'episodes'):
                            episodes = agent_obj.memory.episodes
                            if episodes:
                                st.markdown("#### 📝 Recent Episodes")
                                for ep in episodes[-5:]:
                                    situation = ep.get('situation', 'N/A')[:100]
                                    decision = ep.get('decision', 'N/A')[:80]
                                    success = '✅' if ep.get('outcome', {}).get('success') else '❌'
                                    day = ep.get('day', '?')
                                    st.markdown(f"""
                                    **Day {day}** {success}  
                                    📋 *Situation:* {situation}  
                                    🎯 *Decision:* {decision}
                                    ---
                                    """)
                            else:
                                st.info("No episodes stored yet. Run simulation steps to build memory.")

                        # Show recall example
                        if agent_obj and hasattr(agent_obj.memory, 'recall_similar'):
                            similar = agent_obj.memory.recall_similar("inventory low, demand high", n_results=3)
                            if similar:
                                st.markdown("#### 🔍 Recall: 'inventory low, demand high'")
                                for s in similar:
                                    st.code(str(s)[:200], language='text')
            else:
                st.info("No memory data available yet. Run a few simulation steps!")
        else:
            st.info("Memory not available. Switch to **agentic** mode for memory features.")

    # =================== TAB 12: FORECAST ACCURACY ===================
    with tab12:
        st.header("🎯 Forecast Accuracy — Predicted vs Actual")
        st.markdown("How well is the LSTM model predicting demand?")

        data = st.session_state.data
        if data and len(data) >= 3:
            df = pd.DataFrame(data)

            # Track predictions vs actuals
            actuals = [r.get('demand', 0) for r in data]
            # Get forecaster predictions for comparison
            fc = st.session_state.fc
            predictions = []
            for i, r in enumerate(data):
                try:
                    pred = fc.predict_next()
                    predictions.append(pred)
                except Exception:
                    predictions.append(actuals[i] if i < len(actuals) else 100)

            days_list = [r.get('day', i+1) for i, r in enumerate(data)]

            # Calculate error metrics
            import numpy as np
            actuals_arr = np.array(actuals, dtype=float)
            preds_arr = np.array(predictions[:len(actuals)], dtype=float)
            errors = actuals_arr - preds_arr
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            mape = np.mean(np.abs(errors / np.maximum(actuals_arr, 1))) * 100

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📏 MAE", f"{mae:.1f} units", help="Mean Absolute Error")
            m2.metric("📐 RMSE", f"{rmse:.1f} units", help="Root Mean Squared Error")
            m3.metric("📊 MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error")
            accuracy = max(0, 100 - mape)
            m4.metric("🎯 Accuracy", f"{accuracy:.1f}%",
                      delta=f"{'Good' if accuracy > 85 else 'Needs improvement'}")

            # Predicted vs Actual chart
            st.subheader("📈 Predicted vs Actual Demand")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days_list, y=actuals,
                name='Actual Demand', line=dict(color='#3b82f6', width=3)))
            fig.add_trace(go.Scatter(x=days_list, y=predictions[:len(days_list)],
                name='Predicted', line=dict(color='#f59e0b', width=2, dash='dash')))
            fig.update_layout(height=350, template='plotly_dark',
                xaxis_title="Day", yaxis_title="Demand (units)",
                title="Forecast vs Reality")
            st.plotly_chart(fig, use_container_width=True)

            # Error distribution
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Error Distribution")
                fig_err = go.Figure()
                fig_err.add_trace(go.Histogram(
                    x=errors, nbinsx=20,
                    marker_color='#8b5cf6', opacity=0.8
                ))
                fig_err.add_vline(x=0, line_dash="dash", line_color="#22c55e",
                    annotation_text="Perfect (0 error)")
                fig_err.update_layout(height=300, template='plotly_dark',
                    xaxis_title="Error (Actual - Predicted)",
                    yaxis_title="Frequency")
                st.plotly_chart(fig_err, use_container_width=True)

            with c2:
                st.subheader("📉 Cumulative Error")
                cum_error = np.cumsum(np.abs(errors))
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=days_list, y=cum_error,
                    fill='tozeroy',
                    line=dict(color='#ef4444', width=2)
                ))
                fig_cum.update_layout(height=300, template='plotly_dark',
                    xaxis_title="Day", yaxis_title="Cumulative |Error|")
                st.plotly_chart(fig_cum, use_container_width=True)

            # Forecast accuracy rating
            if accuracy >= 90:
                st.success(f"🏆 **Excellent forecast accuracy!** {accuracy:.1f}% — the LSTM model is performing very well.")
            elif accuracy >= 75:
                st.warning(f"⚠️ **Decent accuracy:** {accuracy:.1f}% — model captures trends but has some variance.")
            else:
                st.error(f"❌ **Low accuracy:** {accuracy:.1f}% — model may need more training data or tuning.")
        else:
            st.info("▶️ Run at least 3 simulation steps to see forecast accuracy!")

    # =================== SIDEBAR ===================
    with st.sidebar:
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
