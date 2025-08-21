# 🤖 AI Assistant for Data Science

## 📌 Project Overview

This project is an AI-powered data science assistant designed to simplify and accelerate the data analysis workflow. Users can upload CSV datasets and interact with them in natural language through a Streamlit interface. Powered by **Cohere LLM** and **LangChain agents**, the assistant automates exploratory data analysis (EDA), reframes business problems into data science tasks, suggests suitable machine learning algorithms, and even generates Python code solutions — acting as a co-pilot for data scientists.

---

## 🚀 Features

* **CSV Upload & Exploration** – Upload datasets and view summaries, statistics, and correlations.
* **Automated EDA** – Detect missing values, duplicates, and outliers with AI support.
* **Conversational Queries** – Ask natural language questions about your data using LangChain’s Pandas agent.
* **Business Problem to Data Science Mapping** – Reframe business challenges into structured ML tasks.
* **ML Algorithm Suggestions** – Get suitable algorithm recommendations with Wikipedia-enhanced knowledge.
* **Python Code Generation** – Automatically generate executable Python code for selected algorithms using LangChain’s Python REPL agent.
* **Interactive Visualizations** – Boxplots, correlation heatmaps, and summary statistics powered by Seaborn & Matplotlib.
* **Robustness** – Built-in caching, session state, and error handling for stability.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend Orchestration:** LangChain (Agents, Sequential Chains)
* **LLM:** Cohere (command-r-plus)
* **Visualization:** Matplotlib, Seaborn
* **Data Handling:** Pandas
* **Utilities:** Wikipedia API Wrapper, dotenv

---

## 📂 Project Workflow

1. **User uploads a CSV file** via the Streamlit UI.
2. **EDA pipeline** runs automatically with missing value checks, duplicates, correlation, and summary statistics.
3. **LangChain Pandas agent** enables natural language queries on the dataset.
4. **LangChain SequentialChain** reframes the user’s business problem into a data science problem and suggests ML algorithms using Wikipedia knowledge.
5. **Python agent with REPL** generates executable code for the chosen algorithm.
6. **Results and insights** are displayed interactively on the Streamlit dashboard.

---

## 📸 Screenshots (Optional)

*Add images of your Streamlit interface here for better presentation.*

---

## ⚡ Installation & Usage

1. **Clone the repository**

```bash
git clone https://github.com/KaushalKshirsagar/AI-Assistant-for-Data-Science.git
cd AI-Assistant-for-Data-Science
```

2. **Create a virtual environment & activate it**

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

* Create a `.env` file and add your API key:

```env
COHERE_API_KEY=your_api_key_here
```

5. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 📖 Example Use Case

* Upload a sales dataset
* Ask: *“What are the steps of EDA?”*
* Get a structured breakdown with visual insights
* Define a business problem like *“Predict future sales”*
* Assistant reframes it into a machine learning task and suggests algorithms
* Generate ready-to-use Python code for model training

---

## 📌 Future Enhancements

* Integration with vector databases (Pinecone/FAISS) for retrieval-augmented generation
* Support for larger datasets via chunking and streaming
* Fine-tuning for domain-specific tasks
* Model performance evaluation and visualization

Would you like me to also prepare a **requirements.txt file content** for your repo, so anyone cloning it can install dependencies instantly?
