import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


from data_layer import DataLayer
from forecasting_module import DemandForecaster
from supply_chain_model import SupplyChainModel
from kpi_evaluator import KPIEvaluator


def main():
    st.set_page_config(
        page_title="Agentic AI Supply Chain",
        page_icon="📦",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {background-color: #0e1117;}
        .stMetric {background-color: #1e2530; padding: 15px; border-radius: 10px;}
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🤖 Agentic AI for Resilient Retail Supply Chains")
    st.markdown("Real-time simulation with LSTM forecasting, LLM agents, and Mesa framework")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.data_layer = DataLayer()
        st.session_state.historical_data = st.session_state.data_layer.generate_historical_data(365)
        
        st.session_state.forecaster = DemandForecaster(method='lstm')
        with st.spinner('Training LSTM model...'):
            st.session_state.forecaster.train(st.session_state.historical_data, epochs=30)
        
        st.session_state.model = SupplyChainModel(
            st.session_state.forecaster,
            st.session_state.data_layer
        )
        st.session_state.kpi_evaluator = KPIEvaluator()
        st.session_state.simulation_data = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Simulation Controls")
        
        if st.button("▶️ Run Simulation Step", use_container_width=True):
            model = st.session_state.model
            model.step()
            
            # Update KPIs
            fulfilled = model.warehouse.inventory if model.daily_demand <= model.warehouse.inventory else model.warehouse.inventory
            stockout = fulfilled < model.daily_demand
            
            st.session_state.kpi_evaluator.update(
                model.daily_demand,
                fulfilled,
                model.warehouse.inventory,
                stockout
            )
            
            st.session_state.simulation_data.append({
                'day': model.current_day,
                'inventory': model.warehouse.inventory,
                'demand': model.daily_demand,
                'fulfilled': fulfilled
            })
        
        if st.button("🔄 Reset Simulation", use_container_width=True):
            st.session_state.model = SupplyChainModel(
                st.session_state.forecaster,
                st.session_state.data_layer
            )
            st.session_state.kpi_evaluator = KPIEvaluator()
            st.session_state.simulation_data = []
        
        st.divider()
        st.header("⚠️ Disruption Testing")
        
        if st.button("Disrupt Supplier", use_container_width=True):
            st.session_state.model.inject_disruption('supplier', 3)
            st.warning("Supplier disrupted!")
        
        if st.button("Disrupt Logistics", use_container_width=True):
            st.session_state.model.inject_disruption('logistics', 3)
            st.warning("Logistics disrupted!")
        
        st.divider()
        st.info(f"📅 Current Day: {st.session_state.model.current_day}")
    
    # KPI Dashboard
    st.header("📊 Key Performance Indicators")
    kpis = st.session_state.kpi_evaluator.calculate_kpis()
    
    cols = st.columns(5)
    kpi_names = ['Fill Rate (%)', 'Stock-out Rate (%)', 'Avg Inventory', 
                 'Resilience Index', 'Customer Satisfaction']
    
    for col, name in zip(cols, kpi_names):
        with col:
            value = kpis.get(name, 0)
            st.metric(name, f"{value}")
    
    # Visualization
    if st.session_state.simulation_data:
        df = pd.DataFrame(st.session_state.simulation_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Inventory & Demand Trends")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=df['day'], y=df['inventory'], name="Inventory",
                          line=dict(color='#3b82f6', width=2)),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=df['day'], y=df['demand'], name="Demand",
                          line=dict(color='#ef4444', width=2)),
                secondary_y=True
            )
            
            fig.update_layout(
                height=400,
                hovermode='x unified',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("✅ Fulfillment Performance")
            fig = go.Figure()
            
            fig.add_trace(go.Bar(x=df['day'], y=df['demand'], name='Demand',
                               marker_color='#ef4444'))
            fig.add_trace(go.Bar(x=df['day'], y=df['fulfilled'], name='Fulfilled',
                               marker_color='#10b981'))
            
            fig.update_layout(
                height=400,
                barmode='group',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Agent Logs
    st.header("📝 Agent Activity Logs")
    if st.session_state.model.event_log:
        recent_logs = st.session_state.model.event_log[-20:]
        for log in reversed(recent_logs):
            level = 'info'
            if 'disrupted' in log['message'].lower() or 'stock-out' in log['message'].lower():
                level = 'error'
            elif 'partial' in log['message'].lower():
                level = 'warning'
            
            if level == 'error':
                st.error(f"Day {log['day']} | {log['agent']}: {log['message']}")
            elif level == 'warning':
                st.warning(f"Day {log['day']} | {log['agent']}: {log['message']}")
            else:
                st.info(f"Day {log['day']} | {log['agent']}: {log['message']}")
    else:
        st.info("No activity logs yet. Run simulation to see agent actions.")

if __name__ == "__main__":
    main()
