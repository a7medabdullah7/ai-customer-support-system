# ==========================================
# IMPORTS
# ==========================================

import base64
import os
import pickle
import time
import datetime
import ollama
import streamlit as st
import pandas as pd
import plotly.express as px

from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ==========================================
# CONFIG
# ==========================================

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
LOG_FILE = "logs.csv"

st.set_page_config(page_title="AI Support Platform", layout="wide")

# ==========================================
# UTILITIES
# ==========================================

def load_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["email","category","sentiment","priority","time"])

def save_logs(df):
    df.to_csv(LOG_FILE, index=False)

def add_log(email, cat, sen, prio):
    df = load_logs()
    df.loc[len(df)] = [email, cat, sen, prio, datetime.datetime.now()]
    save_logs(df)

# ==========================================
# GMAIL AUTH
# ==========================================

@st.cache_resource
def gmail_auth():
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle","rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.pickle","wb") as token:
            pickle.dump(creds,token)

    return build("gmail","v1",credentials=creds)

# ==========================================
# GMAIL FUNCTIONS
# ==========================================

def read_emails(service, limit=10):

    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=limit
    ).execute()

    messages = results.get("messages",[])
    emails = []

    for msg in messages:
        try:
            txt = service.users().messages().get(
                userId="me",
                id=msg["id"],
                format="full"
            ).execute()

            payload = txt["payload"]
            parts = payload.get("parts")

            if parts:
                data = parts[0]["body"]["data"]
                text = base64.urlsafe_b64decode(data).decode(errors="ignore")
                emails.append((msg["id"], text))

        except:
            continue

    return emails

def send_reply(service, msg_id, reply):

    message = MIMEText(reply)
    message["to"] = "your@email.com"
    message["subject"] = "Re: Support Request"

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    service.users().messages().send(
        userId="me",
        body={"raw": raw}
    ).execute()

# ==========================================
# AI ENGINE (UPDATED 🔥)
# ==========================================

def ai_call(prompt):
    try:
        r = ollama.chat(
            model="llama3",
            messages=[{"role":"user","content":prompt}]
        )
        return r["message"]["content"].strip()
    except:
        return "AI Error"

def classify_email(text):
    return ai_call(f"""
Classify this email into ONE category:
Login Issue, Billing, Technical Issue, Complaint, General Inquiry

Email:
{text}
""")

def sentiment(text):
    return ai_call(f"""
Analyze sentiment. Answer ONLY:
Positive or Neutral or Negative

Email:
{text}
""")

# 🔥 UPDATED PRIORITY SYSTEM
def priority(text, sen):
    text = text.lower()
    sen = sen.lower()

    # كلمات خطيرة
    urgent_keywords = ["urgent", "asap", "immediately", "now"]
    angry_keywords = ["refund", "cancel", "angry", "frustrated", "worst", "unacceptable"]

    if any(word in text for word in urgent_keywords):
        return "High"

    if any(word in text for word in angry_keywords):
        return "High"

    if "negative" in sen:
        return "High"

    if "neutral" in sen:
        return "Medium"

    return "Low"

def generate_reply(text):
    return ai_call(f"""
Write a professional customer support reply.
Be polite, empathetic, and solution-oriented.

Email:
{text}
""")

# ==========================================
# AUTO REPLY ENGINE
# ==========================================

def auto_reply(service):

    emails = read_emails(service, limit=5)

    for msg_id, email in emails:

        cat = classify_email(email)
        sen = sentiment(email)
        prio = priority(email, sen)

        reply = generate_reply(email)

        send_reply(service, msg_id, reply)
        add_log(email, cat, sen, prio)

# ==========================================
# DASHBOARD
# ==========================================

def dashboard():

    st.subheader("📊 System Overview")

    df = load_logs()

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Tickets", len(df))
    col2.metric("High Priority", len(df[df["priority"]=="High"]))
    col3.metric("Negative", len(df[df["sentiment"].str.contains("Negative", na=False)]))
    col4.metric("Categories", df["category"].nunique())

    st.divider()

    if st.button("🚀 Run Auto Reply"):
        with st.spinner("Processing..."):
            auto_reply(service)
        st.success("Done")

# ==========================================
# INBOX
# ==========================================

def inbox():

    st.subheader("📥 Inbox")

    emails = read_emails(service, 5)

    for msg_id, email in emails:

        with st.expander("Email"):

            st.write(email[:500])

            if st.button(f"Analyze {msg_id}"):

                cat = classify_email(email)
                sen = sentiment(email)
                prio = priority(email, sen)

                st.write(f"Category: {cat}")
                st.write(f"Sentiment: {sen}")
                st.write(f"Priority: {prio}")

            if st.button(f"Reply {msg_id}"):

                reply = generate_reply(email)

                st.success("Reply Generated")
                st.write(reply)

# ==========================================
# TEST AI
# ==========================================

def test_ai():

    st.subheader("🧪 Test AI")

    email = st.text_area("Enter Email")

    if st.button("Generate"):

        if not email:
            st.warning("Enter text first")
            return

        with st.spinner("Thinking..."):

            cat = classify_email(email)
            sen = sentiment(email)
            prio = priority(email, sen)
            reply = generate_reply(email)

        st.success("Done")

        st.write("### 📌 Analysis")
        st.write(f"Category: {cat}")
        st.write(f"Sentiment: {sen}")
        st.write(f"Priority: {prio}")

        st.write("### 🤖 Reply")
        st.write(reply)

# ==========================================
# ANALYTICS
# ==========================================

def analytics():

    st.subheader("📈 Analytics")

    df = load_logs()

    if df.empty:
        st.info("No data yet")
        return

    col1,col2 = st.columns(2)

    with col1:
        fig = px.pie(df, names="category", title="Categories")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(df["priority"].value_counts(), title="Priority")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "logs.csv"
    )

# ==========================================
# MAIN APP
# ==========================================

st.title("🚀 AI Customer Support Platform PRO")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard","Inbox","Test AI","Analytics"]
)

service = gmail_auth()

if menu == "Dashboard":
    dashboard()

elif menu == "Inbox":
    inbox()

elif menu == "Test AI":
    test_ai()

elif menu == "Analytics":
    analytics()