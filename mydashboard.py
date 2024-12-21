import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Description
st.title("Interactive Dashboard: Synthetic Dataset Analysis")
st.markdown("### A comparison of raw vs preprocessed dataset performance")

# Sidebar for Navigation
st.sidebar.title("Dashboard Navigation")
section = st.sidebar.radio("Go to Section:", ["Dataset Overview", "Visualizations", "Model Metrics"])

# File Upload Section
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if section == "Dataset Overview":
        # Dataset Preview
        st.markdown("### Dataset Overview")
        st.dataframe(df)
        st.write("Dataset Statistics:")
        st.write(df.describe())

        # Column Selection
        st.markdown("### Select Columns to Display")
        selected_columns = st.multiselect("Select columns:", options=df.columns)
        if selected_columns:
            st.write("Selected Columns:")
            st.dataframe(df[selected_columns])

    elif section == "Visualizations":
        st.markdown("### Visualizations")

        # Data Distribution Plot
        st.markdown("#### Data Distribution")
        column = st.selectbox("Select a column for distribution plot:", options=df.select_dtypes(include=["float64", "int64"]).columns)
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)

        # Correlation Matrix
        st.markdown("#### Correlation Matrix")
        if st.checkbox("Show Correlation Matrix"):
            corr_matrix = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    elif section == "Model Metrics":
        st.markdown("### Model Performance Metrics")

        # Interactive Sliders for Metrics
        raw_mse = st.slider("Raw Data MSE", min_value=0.0, max_value=1.0, value=0.4)
        preprocessed_mse = st.slider("Preprocessed Data MSE", min_value=0.0, max_value=1.0, value=0.2)

        # Metrics Chart
        st.markdown("#### MSE Comparison")
        metrics = pd.DataFrame({"Dataset": ["Raw Data", "Preprocessed Data"], "MSE": [raw_mse, preprocessed_mse]})
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Dataset", y="MSE", data=metrics, ax=ax, palette="Blues_d")
        ax.set_title("MSE Comparison")
        st.pyplot(fig)

        # Additional Metrics
        st.markdown("#### Additional Metrics")
        raw_rmse = st.slider("Raw Data RMSE", min_value=0.0, max_value=2.0, value=0.6)
        preprocessed_rmse = st.slider("Preprocessed Data RMSE", min_value=0.0, max_value=2.0, value=0.4)
        st.write(f"Raw Data RMSE: {raw_rmse}")
        st.write(f"Preprocessed Data RMSE: {preprocessed_rmse}")
