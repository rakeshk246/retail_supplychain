---
title: Retail Supply Chain AI Simulator (v5)
app_file: streamlit_app.py
sdk: streamlit
sdk_version: 1.42.0
---

# 🛍️ AI-Powered Supply Chain Simulator

An advanced agentic supply chain simulation built with **LangGraph**, **Groq LLMs**, **ChromaDB**, and **Streamlit**. This project models a retail supply chain (Supplier → Logistics → Warehouse → Customer) where autonomous AI agents negotiate, communicate, and make decisions to fulfill customer demand while minimizing costs during severe market chaos.

![Streamlit UI Concept](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LLM via Groq](https://img.shields.io/badge/LLM-Groq_%7C_Llama_3-f59e0b?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-059669?style=for-the-badge)

🚀 **Live Demo:** [Play with the Agentic Supply Chain Simulator here!](https://retailsupplychain-a2b3hdk5s49ucflvhzc8pk.streamlit.app/)

## 🌟 Key Features

*   **🤖 Multi-Agent Orchestration:** Distinct AI agents (Demand, Warehouse, Logistics, Supplier) executing in a highly structured, turn-based **LangGraph** cycle.
*   **🌍 External Intelligence Engine:** The simulation doesn't operate in a vacuum. It pulls live meteorological data via **Open-Meteo** and live geopolitical disruption reports via **DuckDuckGo News** to mathematically gauge supply chain risk.
*   **🧠 Explainable AI (XAI) Dashboard:** The agents communicate via strict JSON logs that render into distinct Executive Analytic Cards on the Streamlit UI, allowing reviewers to read the Trigger Reason, Numerical Evidence, and Actions taken dynamically.
*   **📚 Vector Memory (ChromaDB):** Agents learn from past successes and failures, retrieving historical episodes dynamically via zero-shot Context RAG before attempting emergency reorders.
*   **📈 Machine Learning Forecasts:** Relies on a TensorFlow/Keras LSTM model trained over the authentic **Walmart M5 forecasting dataset (CA_1)** to predict quantitative demand thresholds.
*   **🌪️ Live Chaos Monkeys:** Includes interactive disruption buttons allowing operators to simulate Hurricanes, Road Blockages, and panic-buy Demand Spikes in real-time.

---

## 📁 Core Code Architecture
Our architecture has been heavily streamlined. The essential files governing the simulation are:
1.  **`streamlit_app.py`**: The central application entry point and HITL (Human-In-The-Loop) interactive dashboard.
2.  **`orchestrator.py`**: The LangGraph engine governing the turn-by-turn state machine and agent invocation logic.
3.  **`agentic_agents.py`**: Defines the Groq LLaMA models powering our supply chain nodes.
4.  **`news_search.py`**: Handles external intelligence gathering (Weather + News correlation).
5.  **`forecasting_module.py`**: Deep learning LSTM module predicting baseline market demand.
6.  **`message_bus.py`**: The JSON payload serialization infrastructure mimicking an enterprise ERP network.

*(Note: In v5, legacy un-orchestrated simulators and experimental React endpoints have been formally deprecated to maintain a highly pristine backend).*

---

## 🚀 Getting Started

Follow these instructions to set up and run the simulation on your local Python environment.

### 1. Prerequisites

You will need the following installed:
*   **Python 3.10+**
*   **Git**

You also need an API key from **Groq** to power the high-speed AI agents. You can get one for free at [console.groq.com](https://console.groq.com/).

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd Retail_supplychain
```

### 3. Set Up a Virtual Environment 

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

### 6. Add the M5 Dataset

For realistic LSTM training, the system expects the actual Walmart M5 dataset. 
1.  Download the **M5 Forecasting - Accuracy** dataset from Kaggle (you need the `sales_train_evaluation.csv` file).
2.  Place the file inside the data directory: `data/sales_train_evaluation.csv`.

*(Note: If the dataset is unfound, the system defaults to generating synthetic approximations).*

### 7. Run the Simulation

Launch the interactive Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🎮 How to Use the Simulator

1.  **Advance Time:** Let the LSTM module initialize, then click the **▶️ Next Day** button to simulate one chronologically orchestrated day, or execute them in 10-day batches.
2.  **Monitor the Map:** Watch operational capacity logic execute across the supply chain tiers on the interactive map.
3.  **Audit the Agents:** Click the **AI Brain & Comms** tab. Track the live Open-Meteo external intelligence risks, and observe the specific analytic structured cards communicating Triggers, Variables, and Actions between the agents.
4.  **Inject Chaos:** Force a rapid supply chain breakdown by injecting a `Hurricane` or `Demand Spike`. Watch as the agents consult their ChromaDB memory to dynamically navigate the failure cascade without hitting total stockouts!
