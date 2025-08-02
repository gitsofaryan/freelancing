import streamlit as st
import os
from summarizer import summarize_files

st.set_page_config(page_title="Code Directory Summarizer", layout="wide")

st.title("Code & Config Directory Summarizer")
st.write("""
This app reads all supported files in a directory and uses a local AI model to generate a high-level summary (2-3 key lines per file).
""")

directory = st.text_input("Enter directory path to summarize:", value="example_dir/")

if st.button("Summarize Directory"):
    if not os.path.isdir(directory):
        st.error(f"Directory '{directory}' not found.")
    else:
        with st.spinner("Analyzing files and generating summary..."):
            summary = summarize_files(directory)
        st.markdown("## Directory Summary")
        st.write(summary)
