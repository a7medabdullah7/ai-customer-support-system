import streamlit as st
import pandas as pd
import ollama
import faiss
import numpy as np
import os

from sentence_transformers import SentenceTransformer


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="AI Email Support Agent",
    page_icon="📧",
    layout="wide"
)

st.title("📧 AI Email Support Agent")


# =========================
# Create logs file if missing
# =========================

if not os.path.exists("logs.csv"):
    df = pd.DataFrame(columns=["email", "category", "score"])
    df.to_csv("logs.csv", index=False)


# =========================
# Load Embedding Model
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# =========================
# Load Vector Database
# =========================

@st.cache_resource
def load_vector_db():

    index = faiss.read_index("vector.index")
    data = pd.read_pickle("emails.pkl")

    return index, data


index, data = load_vector_db()


# =========================
# Email Classification
# =========================

def classify_email(text):

    prompt = f"""
Classify this email into one of these categories:

Login Issue
Billing
Technical Support
Account Problem
General Question

Email:
{text}

Return only the category name.
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()


# =========================
# UI Input
# =========================

query = st.text_area(
    "Enter customer email",
    height=200,
    placeholder="Example: I cannot login to my account after resetting my password."
)


# =========================
# Generate Reply
# =========================

if st.button("Generate Reply") and query:

    # ---------------------
    # Step 1 Semantic Search
    # ---------------------

    with st.spinner("🔎 Searching knowledge base..."):

        query_embedding = model.encode([query])

        D, I = index.search(np.array(query_embedding), k=1)

        best_match_index = I[0][0]

        solution = data.iloc[best_match_index]["answer"]

        score = float(1 / (1 + D[0][0]))

    st.success("Knowledge base match found")

    col1, col2 = st.columns(2)

    col1.metric("Similarity Score", round(score, 3))


    # ---------------------
    # Step 2 Email Classification
    # ---------------------

    with st.spinner("📂 Classifying email..."):

        category = classify_email(query)

    col2.metric("Category", category)


    # ---------------------
    # Step 3 Generate AI Reply
    # ---------------------

    with st.spinner("🤖 Generating AI reply..."):

        prompt = f"""
Customer email:
{query}

Support solution:
{solution}

Write a professional support email reply.
"""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        reply = response["message"]["content"]


    # ---------------------
    # Show Reply
    # ---------------------

    st.subheader("AI Reply")

    st.write(reply)


    # ---------------------
    # Save to Logs
    # ---------------------

    clean_email = query.replace("\n", " ")

    log = {
        "email": clean_email,
        "category": category,
        "score": score
    }

    df = pd.DataFrame([log])

    df.to_csv("logs.csv", mode="a", header=False, index=False)


    # ---------------------
    # Download Reply
    # ---------------------

    st.download_button(
        "Download Reply",
        reply,
        file_name="support_reply.txt"
    )


# =========================
# Footer
# =========================

st.markdown("---")
st.caption("AI Email Support Agent | Local LLM + Vector Search + Dashboard")