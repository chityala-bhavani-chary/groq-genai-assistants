import streamlit as st
import pandas as pd
from groq import Groq
import matplotlib.pyplot as plt
# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Data Analysis AI Assistant",
    page_icon="🤖",
    layout="wide")

# ----------------------------------
# TITLE
# ----------------------------------
st.title("Data Analysis AI Assistant")
st.write("Upload your dataset and ask questions about it!")

# ----------------------------------
# Groq Api key (from environment)
# ----------------------------------
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    st.error("Set environment variable GROQ_API_KEY before running this app.")
    st.code("export GROQ_API_KEY='your_key_here'\nstreamlit run DA_Ai_Assis2.py", language="bash")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# ----------------------------------
# FILE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader(
    "📁Upload your dataset (CSV)", 
    type=["csv"])

# ----------------------------------
# PROCESS UPLOADED FILE
# ----------------------------------
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the DataFrame
    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head())
    # -----------------------------------
    # Missing Values Handling
    # -----------------------------------
    st.subheader("⚠️ Missing Values Handling")
    missing_option = st.selectbox(
        "Select how to handle missing values",
        ["None", "Drop NA Rows", "Fill with Mean (Numerical)","Fill with Median", "Fill with Mode (Categorical)"])
    df_clean = df.copy()
    if missing_option == "Drop NA Rows":
        df_clean = df_clean.dropna()

    elif missing_option == "Fill with Mean (Numerical)":
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))

    elif missing_option == "Fill with Median":
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    elif missing_option == "Fill with Mode (Categorical)":
        for col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    df = df_clean
    # ----------------------------------
    # Clean Dataset Preview
    # ----------------------------------
    st.subheader("👀 Cleaned Dataset Preview")
    st.dataframe(df_clean.head())

    # Basic information about the dataset
    st.subheader("📊 Dataset Information")
    col1, col2, col3  = st.columns(3)

    with col1:
        st.metric("Number of Rows", df.shape[0])

    with col2:
        st.metric("Number of Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # ----------------------------------
    # Dataset summary
    # ----------------------------------
    st.subheader("📈 Dataset Summary")

    # Shape table
    shape_df = pd.DataFrame({
        "Metric": ["Number of Rows", "Number of Columns"],
        "Value": [df.shape[0], df.shape[1]]
    })

    st.write("**📦 Dataset Shape**")
    st.table(shape_df)

    # Column information
    column_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": [df[col].nunique() for col in df.columns]
    })

    st.write("**🔍 Column Information**")
    st.dataframe(column_info)

    # Numerical Statistics summary
    numerical_df = df.select_dtypes(include=['int64', 'float64'])

    if not numerical_df.empty:
        st.write("**📊 Numerical Statistics**")
        st.dataframe(numerical_df.describe().T)

    # Missing Values Analysis
    missing_df = pd.DataFrame({
        "Column Name": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (df.isnull().sum() / len(df) * 100).values
    })

    st.write("**⚠️ Missing Values Analysis**")
    st.dataframe(missing_df)

    # Sample data
    st.write("**🔍 Sample Data**")
    st.dataframe(df.head())

    # ----------------------------------
    # Summary for AI
    # ----------------------------------
    summary = f"""
    Dataset Shape:
    {shape_df.to_string(index=False)}

    Column Information:
    {column_info.to_string(index=False)}

    Numerical Statistics:
    {numerical_df.describe().T.to_string() if not numerical_df.empty else "No numerical columns"}

    Missing Values:
    {missing_df.to_string(index=False)}

    Sample Data:
    {df.head().to_string()}
    """
    # =====================================
    # Suggested Visualizations
    # =====================================
    st.subheader("📈 Suggested Data Visualizations")
    # All columns
    all_cols = df.columns
    selected_col = st.selectbox(
        "Select a Column",
        all_cols)
    # Create figure
    fig = plt.figure(figsize=(10, 5))

    # Numerical Column
    if df[selected_col].dtype in ["int64", "float64"]:
        df[selected_col].hist(bins=30)
        plt.title(f"Distribution of {selected_col}")
        plt.xlabel(selected_col)
        plt.ylabel("Frequency")

    # Categorical Column
    else:
        df[selected_col].value_counts().head(15).plot(
            kind="bar")
        plt.title(f"Distribution of {selected_col}")
        plt.xlabel(selected_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
    st.pyplot(fig)

    # ----------------------------------
    # AI Insights
    # ----------------------------------
    st.subheader("🤖 AI Insights")
    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing dataset using AI..."):
            prompt = f"""
            You are a senior data analyst.
            Analyze the dataset carefully and provide insights based on the following summary:
            {summary}
            Provide insights such as:
            1. Dataset Overview
            2. Key Patterns and Trends
            3. Missing Values Analysis
            4. Business Recommendations
            5. Suggested Visualizations
            6. Important Trends
            """

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }],
                temperature=0.5,
                max_tokens=1500)
            st.session_state["result"] = (response.choices[0].message.content)

    # ----------------------------------
    # Display AI Insights
    # ----------------------------------
    if "result" in st.session_state:

        st.subheader("📊 AI Generated Insights")

        st.write(st.session_state["result"])

    # ----------------------------------
    # Chat with Dataset AI
    # ----------------------------------
    st.subheader("💬 Chat with Dataset AI")
    user_question = st.text_input(
        "Enter your question about the dataset")

    if st.button("Get AI Answer"):
        if user_question.strip():
            with st.spinner("Getting AI answer..."):
                prompt = f"""
                You are a senior data analyst.
                Dataset Summary:
                {summary}
                Question:
                {user_question}
                """
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.5,
                    max_tokens=1000)
                answer = response.choices[0].message.content
                st.success(answer)
        else:
            st.warning(
                "Please enter a question before asking.")