# Detailed Project Report (DPR): AI-Powered Agentic Supply Chain Simulator

## 1. Executive Summary
The **AI-Powered Agentic Supply Chain Simulator** is an advanced, interactive system designed to model, test, and analyze complex retail supply chain dynamics. Moving beyond traditional rule-based heuristics, this project leverages Large Language Models (LLMs) acting as autonomous agents. These agents communicate, negotiate, recall past experiences, and make strategic decisions to balance inventory, minimize costs, and maximize customer fulfillment amidst simulated disruptions. 

The system relies on real-world retail data (Walmart M5 dataset), predictive LSTM forecasting, vector database memory, and LangGraph state-machine orchestration to create an incredibly realistic, explainable, and stress-testable supply chain environment.

---

## 2. Project Objectives
1. **Model Autonomous Decision Making:** Replace rigid "if-then" supply chain logic with adaptive LLM agents that reason about complex trade-offs.
2. **Implement Explainable AI (XAI):** Ensure that every decision made by an AI agent is completely transparent, providing the "why" behind every action.
3. **Simulate Real-World Volatility:** Safely subject the supply chain to demand spikes, supplier failures, and logistics bottlenecks, observing how the AI recovers natively.
4. **Agent Memory & Communication:** Enable agents to learn from past mistakes using vector memory and communicate laterally to negotiate solutions.
5. **Interactive Data Visualization:** Provide a beautifully designed interactive dashboard to track, analyze, and comprehend the flow of goods and AI thought processes.

---

## 3. System Architecture Overview
The system is divided into five distinct layers:

### A. Data Layer
*   **Source:** Uses the Kaggle Walmart M5 `sales_train_evaluation.csv` dataset, extracting real daily sales data.
*   **Functionality:** If the dataset is unavailable, dynamically generates realistic synthetic data with seasonal trends and noise. Feeds the simulation environment with daily scenarios.

### B. Intelligence Layer
*   **Predictive (LSTM):** A TensorFlow/Keras Long Short-Term Memory (LSTM) model trained on historical data provides baseline demand predictions.
*   **Cognitive (LLM):** The Groq API (using models like LLaMA 3.1 8B Instant) powers the autonomous reasoning. It interprets the LSTM baseline alongside current constraints (e.g., supplier down) and outputs structured JSON decisions with reasoning.

### C. Agent Layer
A Mesa-inspired multi-agent system where each node in the supply chain is treated as a distinct "persona":
*   **Demand Agent:** Interprets market trends, adjusts the LSTM forecast based on market sentiment, and generates the final daily customer demand.
*   **Warehouse Agent:** The core inventory manager. Decides when, what, and how much to order from the supplier to avoid stockouts without incurring massive holding costs.
*   **Supplier Agent:** Represents the factory/vendor. Approves or denies warehouse orders based on its current production capacity and disruption status.
*   **Logistics Agent:** Manages transit. Responsible for calculating lead times and officially delivering goods to the warehouse.

### D. Orchestration Layer (LangGraph)
*   **Functionality:** Acts as the "heartbeat" of the simulation. Instead of agents acting randomly, LangGraph enforces a strict sequential DAG (Directed Acyclic Graph) state machine.
*   **Dynamic Routing:** Each day, LangGraph evaluates the supply chain state and routes agents down specific workflow paths:
    *   `NORMAL`: Inventory is healthy, operations run smoothly.
    *   `EMERGENCY`: Inventory is low or a disruption just hit; triggers rapid re-evaluations.
    *   `CRISIS`: Deep stockouts or overlapping disruptions; triggers dramatic AI interventions.

### E. Presentation Layer (Streamlit)
*   Provides a rich, 11-tab user interface for operators to run simulations, view metrics, read AI narratives, and analyze costs.

---

## 4. Core System Mechanisms

### Inter-Agent Communication (Message Bus)
Instead of agents accessing global variables, they use a centralized `MessageBus`. The Warehouse sends an `order_request` to the Supplier. The Supplier responds with an `order_confirmation`. This models real-world corporate silos and forces the LLM agents to rely on acquired information rather than omniscient knowledge.

### Agent Memory (ChromaDB)
When agents make a decision (e.g., "ordered 500 units during a hurricane") and see the outcome ("Fill rate was only 70%"), this "episode" is embedded and saved locally in a ChromaDB vector database.
Before making future decisions, agents query ChromaDB for similar past scenarios. Over time, the agents autonomously figure out optimal safety stock limits for various disruptions without requiring hard-coded updates.

### Explainable AI (XAI) Auditing
Every JSON payload returned by the LLM contains a specific `why` dictionary containing:
*   A plain-english summary of the thought process.
*   Specific contributing factors (e.g., "LSTM forecast trend", "Current pending shipments").
*   Alternatives that were considered but rejected.
This is intercepted and logged into an XAI audit trail, solving the "black box" problem of Agentic AI.

---

## 5. What Has Been Achieved So Far
Currently, the baseline Phase 1, 2, 3, and 4 requirements have been fully successfully integrated and pushed to the repository (`main` branch) and deployed to Streamlit Community Cloud:

1.  **Refactored Foundation:** Cleaned up Mesa deprecations, established robust OOP classes.
2.  **Dataset Integration:** Integrated the Walmart M5 dataset with a robust synthetic fallback.
3.  **LSTM Engine:** Implemented and integrated a functional deep-learning forecasting module.
4.  **Groq LLM Integration:** Connected to lightning-fast Groq endpoints, utilizing LangChain output parsers to force agents to communicate in strict JSON.
5.  **ChromaDB Vector Memory:** Fully functional state persistence; agents recall past episodes.
6.  **LangGraph Orchestration:** Replaced simple loops with adaptive graph state management.
7.  **Complete Streamlit Dashboard:** Built a professional 11-tab UI featuring:
    *   **Day Briefing:** Consolidated supply chain map, agent status, timelines, and raw LLM context.
    *   **Charts & KPIs:** Financial metrics, hold/stockout costs, fill-rate tracking.
    *   **XAI Tab:** Heatmaps, transparency scores, and detailed expandable reasoning steps.
    *   **Architecture & Data Explorer tabs** for educational understanding.
    *   **AI vs. Rules Comparison:** A dedicated stress-test tab proving the ROI of the LLM vs static algorithms.

---

## 6. Future Enhancements & Planned Improvements

While the core system is fully functional, the following improvements are planned for upcoming phases:

### Phase 5: Multi-Product Complexity
*   **Current State:** The system tracks a single aggregate SKU/product.
*   **Enhancement:** Introduce a catalog of 3-5 distinct products (e.g., Perishables, Electronics, Apparel). Each will have distinct demand curves, lead times, storage costs, and shelf lives. Agents will have to optimize basket orders.

### Phase 6: Dynamic Pricing & Supplier Negotiation
*   **Current State:** Costs are fixed and the supplier assumes flat pricing.
*   **Enhancement:** Allow the Supplier agent to introduce dynamic pricing (e.g., surging prices during shortages). Allow the Warehouse agent to *negotiate* bulk discounts or pay premiums for expedited shipping via the Message Bus.

### Phase 7: Multi-Node Network Graph
*   **Current State:** Linear flow (1 Supplier → 1 Logistics → 1 Warehouse).
*   **Enhancement:** Expand to 2 Suppliers (one cheap/slow, one expensive/fast) and 2 Warehouses (regional vs. central). Route optimization agents will have to decide where to fulfill demand from.

### Phase 8: Production Robustness & Scaling
*   **Database Shift:** Migrate the localized episodic logic to a persistent PostgreSQL database with pgvector, allowing cloud-scale concurrent simulations.
*   **Dockerization:** Provide `docker-compose` files to containerize the Streamlit App, Keras/LSTM backend, and databases for enterprise distribution.
*   **Caching mechanisms:** Implement semantic caching (via Redis) to bypass the Groq API for identical, repeated supply chain states to radically reduce token consumption and rate limiting.

---
**End of Report**
