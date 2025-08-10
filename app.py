import os
from apikey import apikey

import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from langchain_community.llms import Cohere
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.utilities import WikipediaAPIWrapper

# ✅ Set Cohere API Key correctly
os.environ['COHERE_API_KEY'] = apikey
load_dotenv(find_dotenv())

# ✅ Initialize Cohere LLM
llm = Cohere(
    model="command-r-plus",
    temperature=0.1,
    max_tokens=2000,
    cohere_api_key=apikey  # Explicit key for safety
)

# Streamlit UI
st.title("AI Assistant for Data Science")
st.write("Hello, I am your AI Assistant to help you throughout your data science projects")

with st.sidebar:
    st.write("*Your Data Science adventure begins with a CSV file.*")
    st.caption('''**You have to upload your CSV file. Once you upload a CSV file, we can start with data exploration and to shape your business challenge into data science framework. I will introduce you coolest ML models and we will use them to tackle your problems.**''')
    st.divider()
    st.caption("<p style='text-align:center'>Made by Kaushal & Kartik</p>", unsafe_allow_html=True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        # ✅ EDA steps from Cohere
        with st.sidebar:
            with st.expander("What are the steps of EDA"):
                try:
                    response = llm.predict("What are the steps of EDA?")
                    st.write(response)
                except Exception as e:
                    st.warning(f"An error occurred: {str(e)}")
                    st.write("Default EDA steps: Data Collection, Cleaning, Exploration, Visualization, and Preprocessing.")

        with st.sidebar:
            with st.expander("Write a paragraph about importance of framing data science problem appropriately."):
                try:
                    response1 = llm.predict("Write a paragraph about importance of framing data science problem appropriately.")
                    st.write(response1)
                except Exception as e:
                    st.warning(f"An error occurred: {str(e)}")
                    st.write()
                

        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            max_iterations=3
        )

        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write(df.head())

            st.write("**Data Cleaning**")
            try:
                columns_df = pandas_agent.invoke({"input": "Analyze each column..."})
                st.write(columns_df.get('output', columns_df) if isinstance(columns_df, dict) else columns_df)
            except Exception as e:
                st.error(f"Error analyzing columns: {str(e)}")

            try:
                missing = pandas_agent.invoke("Count missing values...")
                st.write(missing.get('output', missing) if isinstance(missing, dict) else missing)
            except Exception as e:
                st.error(f"Error checking missing values: {str(e)}")

            try:
                dupes = pandas_agent.invoke("Check for duplicate rows...")
                st.write(dupes.get('output', dupes) if isinstance(dupes, dict) else dupes)
            except Exception as e:
                st.error(f"Error checking duplicates: {str(e)}")

            st.write("**Data Summary**")
            st.write(df.describe())

            try:
                corr = pandas_agent.invoke("Calculate the correlation matrix...")
                st.write(corr.get('output', corr) if isinstance(corr, dict) else corr)
            except Exception as e:
                st.error(f"Error calculating correlation: {str(e)}")

            try:
                outliers = pandas_agent.invoke("Identify outliers...")
                st.write(outliers.get('output', outliers) if isinstance(outliers, dict) else outliers)
            except Exception as e:
                st.error(f"Error identifying outliers: {str(e)}")


        def qn_func(column_name: str):
            if column_name in df.columns:
                fig, ax = plt.subplots()
                sns.boxplot(data=df, y=column_name, ax=ax)
                ax.set_title(f"Box plot of {column_name}")
                st.pyplot(fig)
            else:
                st.warning("Column not found.")


        def qn_df_func():
            try:
                df_info = pandas_agent.invoke(user_question_dataframe)
                st.write(df_info.get('output', df_info) if isinstance(df_info, dict) else df_info)
            except Exception as e:
                st.error(f"Error: {str(e)}")

        @st.cache_data
        def wiki(prompt):
            return WikipediaAPIWrapper().run(prompt)

        @st.cache_data
        def prompt_templated():
            data_problem_template = PromptTemplate(
                input_variables=['business_problem'],
                template='Convert the following business problem into a data science problem: {business_problem}'
            )
            model_selection_template = PromptTemplate(
                input_variables=['data_problem', 'wikipedia_research'],
                template='Give a list of only names of ML algorithms for this problem: {data_problem}, using wiki: {wikipedia_research}'
            )
            return data_problem_template, model_selection_template

        def chains():
            data_problem_chain = LLMChain(llm=llm, prompt=prompt_templated()[0], output_key="data_problem")
            model_selection_chain = LLMChain(llm=llm, prompt=prompt_templated()[1], output_key="model_selection")
            return SequentialChain(
                chains=[data_problem_chain, model_selection_chain],
                input_variables=['business_problem', 'wikipedia_research'],
                output_variables=['data_problem', 'model_selection']
            )

        def chains_output(prompt, wiki_research):
            chain = chains()
            output = chain.invoke({'business_problem': prompt, 'wikipedia_research': wiki_research})
            return output['data_problem'], output['model_selection']

        @st.cache_data
        def list_to_selectbox(my_model_selection_input):
            lines = my_model_selection_input.split('\n')
            algorithms = [line.split(':')[-1].strip() for line in lines if line.strip()]
            algorithms.insert(0, 'Select Algorithm')
            return algorithms

        @st.cache_resource
        def python_agent():
            agent_executor = create_python_agent(
                llm=llm,
                tool = PythonREPLTool(),
                agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors = True
            )
            return agent_executor

        @st.cache_data
        def python_solution(data_problem, selected_algorithm):
            agent = python_agent()
            prompt = (
                f"Write a python script to solve this data science problem: {data_problem}. "
                f"Use the algorithm: {selected_algorithm}. "
                "Return only the code."
            )
            result = agent.invoke({"input": prompt})
            if isinstance(result, dict) and "output" in result:
                return result["output"]
            return result


            
                
        # Run EDA
        st.header("Exploratory Data Analysis")
        st.subheader("General Information about the dataset")
        function_agent()

        # User input
        st.write("Variable of study")
        user_question = st.text_input("What variable are you interested in?")
        if user_question:
            qn_func(user_question)
            st.subheader("Further Study")
            user_question_dataframe = st.text_input("Any question about the dataframe?")
            if user_question_dataframe and user_question_dataframe.lower() != "no":
                qn_df_func()
            if user_question_dataframe and user_question_dataframe.lower() == "no":
                st.write("")

            if user_question_dataframe:
                st.divider()
                st.header("Data Science Problem")
                prompt = st.text_area("What is the business problem?")
                if prompt:
                    wiki_research = wiki(prompt)
                    data_problem, model_selection = chains_output(prompt, wiki_research)
                    st.write(data_problem)
                    st.write(model_selection)

                    formatted_list = list_to_selectbox(model_selection)
                    # ✅ Fixed missing options
                    selected_algorithm = st.selectbox("Select machine learning algorithm", options=formatted_list)
                    if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                        st.subheader("Solution")
                        try:
                            solution_code = python_solution(data_problem, selected_algorithm)
                            if isinstance(solution_code, dict) and "output" in solution_code:
                                solution_code = solution_code["output"]
                            if solution_code:
                                st.code(solution_code, language="python")
                            else:
                                st.warning("No code was returned by the agent.")
                        except Exception as e:
                            st.error(f"Failed to generate solution: {e}")

