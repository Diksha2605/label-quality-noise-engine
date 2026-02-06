import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add src to path
SRC_PATH = os.path.abspath("src")
sys.path.insert(0, SRC_PATH)

from phase8_enhancements.active_relabel import ActiveRelabelQueue
from phase8_enhancements.auto_cleaning import AutoCleaningAdvisor

st.set_page_config(page_title="LQNE Dashboard", layout="wide")

st.title("🧠 LQNE — Label Quality & Noise Evaluation")

# Load data
st.sidebar.header("Upload Trust Report")
uploaded = st.sidebar.file_uploader(
    "Upload sample_trust_report.csv",
    type="csv"
)

if uploaded:
    df = pd.read_csv(uploaded)

    # Controls
    trust_thresh = st.sidebar.slider(
        "Trust Threshold", 0.05, 0.5, 0.25, 0.05
    )
    drop_frac = st.sidebar.slider(
        "Drop Fraction", 0.0, 0.05, 0.01, 0.005
    )
    top_k = st.sidebar.slider(
        "Top-K Relabel Samples", 50, 1000, 300, 50
    )

    # ---- Metrics ----
    st.subheader("📊 Dataset Health")
    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Trust Score", round(df["trust_score"].mean(), 3))
    col2.metric(
        "Low Trust %",
        round((df["trust_score"] < trust_thresh).mean() * 100, 2),
    )
    col3.metric("Total Samples", len(df))

    # ---- Trust Distribution ----
    st.subheader("📈 Trust Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["trust_score"], bins=30)
    ax.set_xlabel("Trust Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # ---- Active Relabeling ----
    st.subheader("🚨 Active Relabeling Queue")
    relabeler = ActiveRelabelQueue(min_priority=0.05)
    relabel_df = relabeler.generate(df, top_k=top_k)
    st.dataframe(relabel_df)

    # ---- Cleaning Recommendations ----
    st.subheader("🧹 Cleaning Recommendations")
    cleaner = AutoCleaningAdvisor(
        trust_threshold=trust_thresh,
        drop_fraction=drop_frac,
    )
    cleaning = cleaner.generate(df)
    st.json(cleaning)

else:
    st.info("👈 Upload `sample_trust_report.csv` to start")
