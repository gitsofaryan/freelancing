import streamlit as st
import os
from collections import defaultdict
from transformers import pipeline, AutoTokenizer
import PyPDF2
import io

# Page config
st.set_page_config(
    page_title="AI Code & Document Analyzer", 
    page_icon="🤖", 
    layout="wide"
)

# CSS for better UI
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #007ACC;
    background-color: #f8f9fa;
}
.user-message {
    border-left-color: #28a745;
    background-color: #d4edda;
}
.bot-message {
    border-left-color: #007ACC;
    background-color: #cce7ff;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# Header
st.title("🤖 AI Code & Document Analyzer")
st.markdown(f"""
**Current User:** `gitsofaryan`  
**Date:** 2025-08-02 14:10:21 UTC  
**AI Models:** Local CPU-based (Streamlit only)
""")

# Code Summarizer Functions
FILE_EXTS = {".py", ".sql", ".yml", ".yaml", ".xml", ".conf", ".ini", ".txt"}
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

@st.cache_resource
def load_summarizer():
    try:
        summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return summarizer, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

def get_files_in_directory(directory):
    file_paths = []
    file_counts = defaultdict(int)
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in FILE_EXTS:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                file_counts[ext] += 1
    
    return file_paths, file_counts

def chunk_text(text, tokenizer, max_tokens=1020):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start = end
    return chunks

def summarize_content(content, summarizer, tokenizer):
    if not content.strip():
        return "File is empty or unreadable."
    
    chunks = chunk_text(content, tokenizer)
    summaries = []
    
    for chunk in chunks:
        try:
            result = summarizer(chunk)
            summaries.append(result[0]['summary_text'])
            if len(summaries) >= 2:
                break
        except Exception as e:
            continue
    
    if not summaries:
        return "Could not generate summary for this file."
    
    return " ".join(summaries)

def create_consolidated_summary(file_counts, individual_summaries):
    overview_parts = []
    for ext, count in file_counts.items():
        if ext == ".py":
            overview_parts.append(f"{count} Python scripts")
        elif ext in [".yml", ".yaml"]:
            overview_parts.append(f"{count} YAML files")
        elif ext == ".sql":
            overview_parts.append(f"{count} SQL files")
        elif ext == ".xml":
            overview_parts.append(f"{count} XML files")
        elif ext in [".conf", ".ini"]:
            overview_parts.append(f"{count} configuration files")
        else:
            overview_parts.append(f"{count} {ext[1:].upper()} files")
    
    if overview_parts:
        overview = f"**📊 Directory Overview:** This directory contains {', '.join(overview_parts)}.\n\n"
        return f"{overview}{individual_summaries}"
    else:
        return individual_summaries

def summarize_files(directory):
    summarizer, tokenizer = load_summarizer()
    if not summarizer:
        return "Error: Could not load AI model."
    
    files, file_counts = get_files_in_directory(directory)
    summary_lines = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file_path in enumerate(files):
        file_name = os.path.relpath(file_path, directory)
        status_text.text(f"Processing: {file_name}")
        
        summary = summarize_content(read_file(file_path), summarizer, tokenizer)
        
        ext = os.path.splitext(file_name)[-1].lower()
        if ext == ".sql":
            summary_lines.append(f"**{file_name}** (SQL): {summary}")
        elif ext == ".py":
            summary_lines.append(f"**{file_name}** (Python): {summary}")
        elif ext in [".yml", ".yaml"]:
            summary_lines.append(f"**{file_name}** (YAML): {summary}")
        elif ext == ".xml":
            summary_lines.append(f"**{file_name}** (XML): {summary}")
        else:
            summary_lines.append(f"**{file_name}**: {summary}")
        
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text("Analysis complete!")
    progress_bar.empty()
    status_text.empty()
    
    if not summary_lines:
        return "No supported files found in the directory."
    
    individual_summaries = "\n\n".join(summary_lines)
    return create_consolidated_summary(file_counts, individual_summaries)

def summarize_uploaded_files(uploaded_files):
    summarizer, tokenizer = load_summarizer()
    if not summarizer:
        return "Error: Could not load AI model."
    
    summary_lines = []
    file_counts = defaultdict(int)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing: {file.name}")
        
        try:
            content = file.read().decode("utf-8", errors="ignore")
            file.seek(0)
        except Exception as e:
            summary_lines.append(f"**{file.name}**: Error reading file: {e}")
            continue
        
        ext = os.path.splitext(file.name)[-1].lower()
        file_counts[ext] += 1
        
        summary = summarize_content(content, summarizer, tokenizer)
        
        if ext == ".sql":
            summary_lines.append(f"**{file.name}** (SQL): {summary}")
        elif ext == ".py":
            summary_lines.append(f"**{file.name}** (Python): {summary}")
        elif ext in [".yml", ".yaml"]:
            summary_lines.append(f"**{file.name}** (YAML): {summary}")
        elif ext == ".xml":
            summary_lines.append(f"**{file.name}** (XML): {summary}")
        else:
            summary_lines.append(f"**{file.name}**: {summary}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Analysis complete!")
    progress_bar.empty()
    status_text.empty()
    
    if not summary_lines:
        return "No uploaded files to summarize."
    
    individual_summaries = "\n\n".join(summary_lines)
    return create_consolidated_summary(file_counts, individual_summaries)

# PDF Functions
def extract_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {e}")
    return text

def answer_pdf_question(question, pdf_text):
    summarizer, tokenizer = load_summarizer()
    if not summarizer:
        return "Error: Could not load AI model."
    
    # Simple Q&A by combining question with relevant text chunks
    if not pdf_text.strip():
        return "No PDF text available. Please upload and process PDFs first."
    
    # Find relevant sections (simple keyword matching)
    question_words = question.lower().split()
    sentences = pdf_text.split('. ')
    relevant_sentences = []
    
    for sentence in sentences:
        if any(word in sentence.lower() for word in question_words):
            relevant_sentences.append(sentence)
            if len(relevant_sentences) >= 5:  # Limit context
                break
    
    if not relevant_sentences:
        relevant_sentences = sentences[:5]  # Use first 5 sentences if no matches
    
    context = '. '.join(relevant_sentences)
    
    # Create a prompt for summarization
    prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
    
    try:
        chunks = chunk_text(prompt, tokenizer, max_tokens=800)
        result = summarizer(chunks[0])
        return result[0]['summary_text']
    except Exception as e:
        return f"Error generating answer: {e}"

# Main App Layout
tab1, tab2 = st.tabs(["📁 Code Directory Analyzer", "📚 PDF Document Chat"])

# Tab 1: Code Directory Summarizer
with tab1:
    st.header("📁 Code & Config Directory Summarizer")
    st.markdown("Analyze code directories and files using local AI models (CPU-based)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        directory = st.text_input("📂 Enter directory path to summarize:", value="")
        uploaded_files = st.file_uploader(
            "📎 Or upload files to summarize", 
            type=['py', 'sql', 'yml', 'yaml', 'xml', 'conf', 'ini', 'txt'], 
            accept_multiple_files=True,
            key="code_files"
        )
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>📋 Supported Files</h4>
        • Python (.py)<br>
        • SQL (.sql)<br>
        • YAML (.yml, .yaml)<br>
        • XML (.xml)<br>
        • Config (.conf, .ini)<br>
        • Text (.txt)
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Code", type="primary", use_container_width=True):
        summaries = []
        
        if directory and os.path.isdir(directory):
            with st.spinner("🔄 Analyzing files in directory..."):
                summaries.append(summarize_files(directory))
        
        if uploaded_files:
            with st.spinner("🔄 Analyzing uploaded files..."):
                summaries.append(summarize_uploaded_files(uploaded_files))
        
        if summaries:
            st.markdown("## 📊 Analysis Results")
            for s in summaries:
                st.markdown(s)
        else:
            st.warning("⚠️ No valid files found or uploaded.")

# Tab 2: PDF Chat
with tab2:
    st.header("📚 PDF Document Chat Assistant")
    st.markdown("Upload PDFs and ask questions about their content (Streamlit-only version)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # PDF Upload
        pdf_files = st.file_uploader(
            "📄 Upload PDF documents", 
            type=['pdf'], 
            accept_multiple_files=True,
            key="pdf_files"
        )
        
        if st.button("🚀 Process PDFs", type="primary"):
            if pdf_files:
                with st.spinner("📖 Extracting text from PDFs..."):
                    st.session_state.pdf_text = extract_pdf_text(pdf_files)
                    st.success(f"✅ Successfully processed {len(pdf_files)} PDF(s)!")
                    st.info(f"📊 Extracted {len(st.session_state.pdf_text)} characters of text")
            else:
                st.warning("⚠️ Please upload PDF files first!")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>💡 How to Use</h4>
        1. Upload PDF files<br>
        2. Click "Process PDFs"<br>
        3. Ask questions below<br>
        4. Get AI-powered answers
        </div>
        """, unsafe_allow_html=True)
    
    # Chat Interface
    if st.session_state.pdf_text:
        st.markdown("### 💬 Ask Questions About Your Documents")
        
        question = st.text_input("❓ Your question:", key="pdf_question")
        
        if st.button("🤖 Get Answer", type="secondary"):
            if question:
                with st.spinner("🧠 Generating answer..."):
                    answer = answer_pdf_question(question, st.session_state.pdf_text)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": question, "answer": answer})
                    
                    # Display answer
                    st.markdown(f"""
                    <div class="chat-message user-message">
                    <strong>🙋 You:</strong> {question}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                    <strong>🤖 AI:</strong> {answer}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 📜 Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                st.markdown(f"""
                <div class="chat-message user-message">
                <strong>🙋 You:</strong> {chat['question']}
                </div>
                <div class="chat-message bot-message">
                <strong>🤖 AI:</strong> {chat['answer']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📝 Upload and process PDF documents to start chatting!")

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ App Info")
    st.markdown(f"""
    **🔧 Technology Stack:**
    - Streamlit UI
    - Transformers (Local AI)
    - PyPDF2 (PDF processing)
    - CPU-based models only
    
    **🎯 Features:**
    - 📁 Code directory analysis
    - 📚 PDF document chat
    - 🤖 Local AI processing
    - 💾 Chat history
    """)
    
    if st.button("🔄 Clear All Data"):
        st.session_state.chat_history = []
        st.session_state.pdf_text = ""
        st.success("✅ All data cleared!")
    
    st.markdown("---")
    st.markdown("**👨‍💻 Developer:** gitsofaryan")
    st.markdown("**📅 Version:** 2025-08-02")
