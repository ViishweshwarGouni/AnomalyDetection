import streamlit as st

st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🚨",
    layout="wide"
)

# ---------------------------------
# Custom Styling
# ---------------------------------

st.markdown("""
<style>

.title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#ef4444;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#64748b;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🚨 Anomaly Detection Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Isolation Forest using Iris Dataset</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🚨 Isolation Forest"
    ]
)

# ---------------------------------
# Home Page
# ---------------------------------

if page == "🏠 Home":

    st.header("Welcome")

    st.info("""
    This project demonstrates anomaly detection
    using the Isolation Forest Algorithm.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset", "Iris")
    col2.metric("Samples", "150")
    col3.metric("Features", "4")

    st.divider()

    st.markdown("""
    ### 🎯 Objectives

    - Detect unusual observations
    - Identify anomalies and outliers
    - Visualize detected anomalies
    - Understand Isolation Forest
    """)

# ---------------------------------
# Isolation Forest Page
# ---------------------------------

elif page == "🚨 Isolation Forest":

    from modules.isolation_forest import run
    run()

st.divider()

st.caption(
    "Anomaly Detection Project | Streamlit + Scikit-Learn"
)
