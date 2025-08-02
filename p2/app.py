import streamlit as st
import os
from summarizer import summarize_files, summarize_uploaded_files

st.set_page_config(page_title="Code Directory Summarizer", layout="wide")

st.title("Code & Config Directory Summarizer")
st.write("""
This app reads all supported files in a directory or from uploads and uses a local AI model to generate a high-level summary (2-3 key lines per file).
""")

directory = st.text_input("Enter directory path to summarize:", value="")
uploaded_files = st.file_uploader(
    "Or upload files to summarize", 
    type=['py', 'sql', 'yml', 'yaml', 'xml', 'conf', 'ini', 'txt'], 
    accept_multiple_files=True
)

if st.button("Summarize"):
    summaries = []
    if directory and os.path.isdir(directory):
        with st.spinner("Analyzing files in directory..."):
            summaries.append(summarize_files(directory))
    if uploaded_files:
        with st.spinner("Analyzing uploaded files..."):
            summaries.append(summarize_uploaded_files(uploaded_files))
    if summaries:
        st.markdown("## Consolidated Summary")
        for s in summaries:
            st.write(s)
    else:
        st.warning("No valid files found or uploaded.")