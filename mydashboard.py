# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets from GitHub repository
preprocessed_data_path = "https://raw.githubusercontent.com/GhiX404/Airline-Ticket-prediction/refs/heads/main/air_ticket_data_preprocessed.csv"
raw_data_path = "https://raw.githubusercontent.com/GhiX404/Airline-Ticket-prediction/refs/heads/main/air_ticket_data.csv"

# Read raw and preprocessed data
try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and Description
st.title("Airline Ticket Price Prediction")
st.markdown("### Comparison of Raw and Preprocessed Dataset Performance")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose Section:", ["Dataset Overview", "Visualizations", "Model Metrics"])

# Tabs for navigation
if section == "Dataset Overview":
    st.header("Dataset Overview")

    # Raw Dataset
    st.subheader("Raw Dataset")
    st.dataframe(df_raw)
    st.write("Summary Statistics for Raw Dataset:")
    st.write(df_raw.describe())

    # Preprocessed Dataset
    st.subheader("Preprocessed Dataset")
    st.dataframe(df_preprocessed)
    st.write("Summary Statistics for Preprocessed Dataset:")
    st.write(df_preprocessed.describe())

elif section == "Visualizations":
    st.header("Visualizations")

    # Data Distribution
    st.subheader("Data Distribution")
    col_to_plot = st.selectbox("Select a column to plot distribution:", df_raw.select_dtypes(include=["float64", "int64"]).columns)

    st.markdown("#### Raw Dataset Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_raw[col_to_plot], kde=True, bins=30, ax=ax, color="blue")
    plt.title(f"{col_to_plot} Distribution (Raw Dataset)")
    st.pyplot(fig)

    st.markdown("#### Preprocessed Dataset Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_preprocessed[col_to_plot], kde=True, bins=30, ax=ax, color="green")
    plt.title(f"{col_to_plot} Distribution (Preprocessed Dataset)")
    st.pyplot(fig)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    dataset_choice = st.radio("Choose dataset:", ["Raw Dataset", "Preprocessed Dataset"])
    if dataset_choice == "Raw Dataset":
        corr = df_raw.select_dtypes(include=["float64", "int64"]).corr()
    else:
        corr = df_preprocessed.select_dtypes(include=["float64", "int64"]).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.title(f"Correlation Matrix ({dataset_choice})")
    st.pyplot(fig)

elif section == "Model Metrics":
    st.header("Model Metrics")

    # Predefined metrics
    metrics = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "R^2"],
        "Raw Dataset": [ 1045524.8117988207, 1022.5090766339538, 795.7220529470528, -0.9329847546589642],
        "Preprocessed Dataset": [0.10454906421171427,0.32334047722441783, 0.1379034503447631, 0.8973534285135758]
    })

    # Separate graphs for each metric
    for metric in metrics["Metric"]:
        st.markdown(f"#### {metric} Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        metric_data = metrics.set_index("Metric").loc[metric]
        metric_data.plot(kind="bar", ax=ax, color=["blue", "green"])
        plt.title(f"{metric} Comparison")
        plt.ylabel(metric)
        plt.xticks(rotation=0)
        st.pyplot(fig)
        print(Raw = , metrics["Raw Dataset"])
        print(Pre-processed = , metrics["Preprocessed Dataset"])
