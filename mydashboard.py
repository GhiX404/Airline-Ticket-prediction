# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

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
st.title("Interactive Dashboard: Airline Ticket Price Prediction")
st.markdown("### Compare Raw and Preprocessed Dataset Performance")

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

    # Automatically calculate metrics from existing data
    try:
        raw_y_true = df_raw["Ticket Price (USD)"]  # Assuming "Ticket Price (USD)" is the target column
        raw_y_pred = df_raw["Predicted Price"]  # Assuming "Predicted Price" exists in the raw dataset

        pre_y_true = df_preprocessed["Ticket Price (USD)"]
        pre_y_pred = df_preprocessed["Predicted Price"]

        raw_mse = mean_squared_error(raw_y_true, raw_y_pred)
        raw_rmse = np.sqrt(raw_mse)
        raw_mae = mean_absolute_error(raw_y_true, raw_y_pred)

        pre_mse = mean_squared_error(pre_y_true, pre_y_pred)
        pre_rmse = np.sqrt(pre_mse)
        pre_mae = mean_absolute_error(pre_y_true, pre_y_pred)

        # Metrics Comparison
        metrics = pd.DataFrame({
            "Metric": ["MSE", "RMSE", "MAE"],
            "Raw Dataset": [raw_mse, raw_rmse, raw_mae],
            "Preprocessed Dataset": [pre_mse, pre_rmse, pre_mae]
        })

        st.markdown("#### Metrics Comparison")
        fig, ax = plt.subplots(figsize=(8, 6))
        metrics.set_index("Metric").plot(kind="bar", ax=ax)
        plt.title("Comparison of Metrics")
        plt.ylabel("Value")
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"KeyError: {e}. Ensure the required columns exist in the datasets.")