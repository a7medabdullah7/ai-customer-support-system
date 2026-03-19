import streamlit as st
import pandas as pd
import os
import plotly.express as px

# Page settings
st.set_page_config(
    page_title="AI Email Support Dashboard",
    layout="wide"
)

# Title
st.title("📧 AI Customer Support Dashboard")
st.markdown("Automated email support system powered by AI")

# Create logs file if not exists
if not os.path.exists("logs.csv"):
    df = pd.DataFrame(columns=["email","category","score"])
    df.to_csv("logs.csv", index=False)

data = pd.read_csv("logs.csv")

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Emails Processed", len(data))

if len(data) > 0:
    col2.metric("Average AI Confidence", round(data["score"].mean(),2))
    col3.metric("Detected Categories", data["category"].nunique())
else:
    col2.metric("Average AI Confidence", 0)
    col3.metric("Detected Categories", 0)

st.divider()

# Recent Emails Table
st.subheader("📨 Recent Support Emails")

st.dataframe(
    data.tail(10),
    use_container_width=True
)

st.divider()

# Charts
if len(data) > 0:

    col1, col2 = st.columns(2)

    # Pie Chart
    with col1:
        fig = px.pie(
            data,
            names="category",
            title="Support Request Categories"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bar Chart
    with col2:
        fig2 = px.bar(
            data["category"].value_counts(),
            title="Email Distribution by Category"
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("No email data available yet.")