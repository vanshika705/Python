import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="KDD Cup Anomaly Detection", layout="wide")

st.title("🚨 KDD Cup 1999 Anomaly Detection Dashboard")

# Load processed data
try:
    df = pd.read_csv("processed_kdd.csv")
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("❌ File 'processed_kdd.csv' not found. Please run the Jupyter Notebook first.")
    st.stop()

# Show dataset preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# Dataset info
st.write("📊 Dataset Shape:", df.shape)

# Anomaly stats
st.subheader("📌 Anomaly Summary")
anomaly_count = df['anomaly'].value_counts()
st.write("✅ Normal:", anomaly_count.get(0, 0))
st.write("🚨 Anomalies:", anomaly_count.get(1, 0))

# Plot anomaly distribution
st.subheader("📈 Anomaly Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='anomaly', palette='Set2', ax=ax)
ax.set_xticklabels(['Normal (0)', 'Anomaly (1)'])
st.pyplot(fig)

# Optional: Protocol vs Anomaly (if protocol_type exists)
if 'protocol_type' in df.columns:
    st.subheader("🔄 Anomalies by Protocol Type")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x='protocol_type', hue='anomaly', palette='cool', ax=ax2)
    st.pyplot(fig2)

st.caption("Built with Streamlit • KDD 1999 Anomaly Detection using Isolation Forest")



