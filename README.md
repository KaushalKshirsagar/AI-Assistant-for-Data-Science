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


<img width="1920" height="1080" alt="Screenshot (155)" src="https://github.com/user-attachments/assets/b4a552a4-7abe-4672-baf1-64c23bec3e71" />

<img width="1920" height="1080" alt="Screenshot (156)" src="https://github.com/user-attachments/assets/923bb2ba-b70c-4c65-8e22-5f79d6c592f9" />

<img width="1920" height="1080" alt="Screenshot (157)" src="https://github.com/user-attachments/assets/7668a35a-1f1a-4183-8860-90e6d7616f88" />

<img width="1920" height="1080" alt="Screenshot (158)" src="https://github.com/user-attachments/assets/819be9c4-defc-4765-aedd-430eb7978a6f" />

<img width="1920" height="1080" alt="Screenshot (159)" src="https://github.com/user-attachments/assets/3ea0adda-8257-433d-ad41-69632691f4b0" />

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
