---
title: Retail Supply Chain AI Simulator
app_file: streamlit_app.py
sdk: streamlit
sdk_version: 1.42.0
---

# 🛍️ AI-Powered Supply Chain Simulator

An advanced agentic supply chain simulation built with **LangGraph**, **Groq LLMs**, **ChromaDB**, and **Streamlit**. This project models a retail supply chain (Supplier → Logistics → Warehouse → Customer) where autonomous AI agents negotiate, communicate, and make decisions to fulfill customer demand while minimizing costs.

![Streamlit UI Concept](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LLM via Groq](https://img.shields.io/badge/LLM-Groq_%7C_Llama_3-f59e0b?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-059669?style=for-the-badge)

🚀 **Live Demo:** [Play with the Agentic Supply Chain Simulator here!](https://retailsupplychain-a2b3hdk5s49ucflvhzc8pk.streamlit.app/)

## 🌟 Key Features

*   **🤖 Multi-Agent System:** Distinct AI agents (Demand, Warehouse, Logistics, Supplier) with specialized roles.
*   **🧠 Explainable AI (XAI):** See exactly *what* each agent is thinking and *why* it made a specific decision.
*   **📚 Vector Memory (ChromaDB):** Agents learn from past successes and failures, recalling past supply chain disruptions to make better future decisions.
*   **📈 LSTM Forecasting:** Uses a TensorFlow/Keras LSTM model trained on the actual **Walmart M5 forecasting dataset** to predict demand.
*   **🔀 LangGraph Workflow:** Dynamically routes daily operations into "Normal", "Emergency", or "Crisis" paths depending on the system's state.
*   **📊 Rich Analytics Dashboard:** 11-tab Streamlit dashboard with real-time tracking of inventory, transit shipments, communication bus, costs, and key performance indicators.

## 📁 System Architecture
1.  **Data Layer:** Ingests the Walmart M5 `sales_train_evaluation.csv` dataset (or generates synthetic data).
2.  **Intelligence Layer:** LSTM model for baseline forecasting; Groq-powered LLMs for strategic decision-making.
3.  **Agent Layer:** Mesa-inspired agents that use the Message Bus for communication.
4.  **Orchestration Layer:** LangGraph state machine controls the daily supply chain heartbeat.
5.  **Presentation Layer:** Streamlit dashboard for interaction and visualization.

---

## 🚀 Getting Started

Follow these instructions to set up and run the simulation on your local machine.

### 1. Prerequisites

You will need the following installed:
*   **Python 3.9+**
*   **Git**

You also need an API key from **Groq** to power the AI agents. You can get one for free at [console.groq.com](https://console.groq.com/).

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd Retail_supplychain
```

### 3. Set Up a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install the required Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

1.  Copy the example environment file to create your own configuration:
    ```bash
    cp .env.example .env
    ```
2.  Open the newly created `.env` file and paste your Groq API key:
    ```env
    GROQ_API_KEY=your_actual_groq_api_key_here
    GROQ_MODEL=llama-3.1-8b-instant
    ```

### 6. (Optional/Recommended) Add the M5 Dataset

For the most realistic simulation, the system expects the actual Walmart M5 dataset. If it cannot find it, it will fall back to generating synthetic data.

1.  Download the **M5 Forecasting - Accuracy** dataset from Kaggle (you need the `sales_train_evaluation.csv` file).
2.  Place the file inside the data directory: `data/sales_train_evaluation.csv`.

*(Note: This file is ignored by git because it is over 100MB)*

### 7. Run the Simulation

Launch the interactive Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🎮 How to Use the Simulator

1.  **Select a Mode:** Choose between **Agentic (LLM)** for full AI, **Rule-Based** for simple heuristics, or **Hybrid**.
2.  **Advance Time:** Click the **▶️ Next Day** button to simulate one day of supply chain operations, or specify a number of days to run in batches.
3.  **Inject Disruptions:** Use the sidebar to trigger Supplier Delays, Delivery Delays, or Demand Spikes to see how the agents react and recover.
4.  **Explore the Tabs:**
    *   **📖 Day Briefing:** Full summary of what happened today, including agent actions, shipments, and the LSTM forecast.
    *   **🧠 Why? (XAI):** Expandable history showing the exact reasoning, contributing factors, and confidence behind every LLM decision.
    *   **📡 Agent Comms:** View real-time messages sent between agents (e.g., Warehouse negotiating with Supplier).
    *   **🧠 Memory:** Browse the ChromaDB vector storage to see what scenarios the agents have learned.
    *   **⚔️ AI vs Rules:** Run a stress-test scenario comparing the LLM agents against standard logic side-by-side.

---

## 🧪 Running Tests

The project includes a robust suite of unit tests. You can run them using `pytest` from the root directory:

```bash
pytest
```
