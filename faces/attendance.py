"""
Attendance Dashboard
----------------------
Streamlit app that reads attendance_log.csv (produced by recognize_and_log.py)
and shows a simple, demo-able dashboard.

Usage:
    streamlit run attendance_dashboard.py
"""

import streamlit as st
import pandas as pd
import os

CSV_PATH = "attendance_log.csv"

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("Face Recognition Attendance Dashboard")

if not os.path.exists(CSV_PATH):
    st.warning("No attendance data yet. Run `recognize_and_log.py` first to generate logs.")
    st.stop()

df = pd.read_csv(CSV_PATH)

if df.empty:
    st.info("Attendance log is empty so far — no faces logged yet.")
    st.stop()

# ---------- Top metrics ----------
col1, col2, col3 = st.columns(3)
col1.metric("Total check-ins", len(df))
col2.metric("Unique people", df["name"].nunique())
col3.metric("Days recorded", df["date"].nunique())

st.divider()

# ---------- Filters ----------
st.subheader("Filter")
people = st.multiselect("Select people", options=sorted(df["name"].unique()), default=list(df["name"].unique()))
filtered_df = df[df["name"].isin(people)]

# ---------- Table ----------
st.subheader("Attendance Log")
st.dataframe(filtered_df.sort_values(by=["date", "time"], ascending=False), use_container_width=True)

# ---------- Chart: check-ins per person ----------
st.subheader("Check-ins per Person")
counts = filtered_df["name"].value_counts()
st.bar_chart(counts)

# ---------- Chart: check-ins per day ----------
st.subheader("Check-ins per Day")
daily_counts = filtered_df.groupby("date").size()
st.line_chart(daily_counts)

# ---------- Download ----------
st.download_button(
    label="Download filtered log as CSV",
    data=filtered_df.to_csv(index=False),
    file_name="attendance_filtered.csv",
    mime="text/csv",
)