import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest


def run():

    st.header("🚨 Isolation Forest")

    st.markdown("""
    Isolation Forest detects anomalies by isolating
    observations through random partitioning.
    """)

    iris = load_iris()
    X = iris.data

    contamination = st.slider(
        "Contamination Rate",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    with st.spinner("Detecting anomalies..."):

        model = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        predictions = model.fit_predict(X)

    anomaly_count = (predictions == -1).sum()
    normal_count = (predictions == 1).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Normal Points",
            normal_count
        )

    with col2:
        st.metric(
            "Anomalies",
            anomaly_count
        )

    st.divider()

    st.subheader("📊 Anomaly Visualization")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        X[predictions == 1, 0],
        X[predictions == 1, 1],
        label="Normal"
    )

    ax.scatter(
        X[predictions == -1, 0],
        X[predictions == -1, 1],
        marker="x",
        s=100,
        label="Anomaly"
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Isolation Forest Results")

    ax.legend()

    st.pyplot(fig)

    st.divider()

    st.subheader("📋 Prediction Results")

    result_df = pd.DataFrame(
        X,
        columns=iris.feature_names
    )

    result_df["Prediction"] = predictions

    st.dataframe(
        result_df.head(20),
        use_container_width=True
    )

    st.info("""
    Prediction Values:

    • 1 → Normal Observation

    • -1 → Anomaly / Outlier
    """)