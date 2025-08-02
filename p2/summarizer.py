import os
from transformers import pipeline, AutoTokenizer

FILE_EXTS = {".py", ".sql", ".yml", ".yaml", ".xml", ".conf", ".ini", ".txt"}
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

def get_files_in_directory(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in FILE_EXTS:
                file_paths.append(os.path.join(root, file))
    return file_paths

def chunk_text(text, tokenizer, max_tokens=1020):
    """Chunk text by actual token count to ensure it fits within model limits"""
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
    
    # Chunk the content to ensure each piece fits within model limits
    chunks = chunk_text(content, tokenizer)
    summaries = []
    
    for chunk in chunks:
        try:
            result = summarizer(chunk)
            summaries.append(result[0]['summary_text'])
            if len(summaries) >= 3:  # Limit to 2-3 summaries per file
                break
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            continue
    
    if not summaries:
        return "Could not generate summary for this file."
    
    return " ".join(summaries)

def summarize_file(file_path, summarizer, tokenizer):
    content = read_file(file_path)
    return summarize_content(content, summarizer, tokenizer)

def summarize_files(directory):
    try:
        summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        return f"Error loading model: {e}"
    
    files = get_files_in_directory(directory)
    summary_lines = []
    
    for file_path in files:
        file_name = os.path.relpath(file_path, directory)
        summary = summarize_file(file_path, summarizer, tokenizer)
        summary_lines.append(f"**{file_name}**: {summary}")
    
    if not summary_lines:
        return "No supported files found in the directory."
    return "\n\n".join(summary_lines)

def summarize_uploaded_files(uploaded_files):
    try:
        summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        return f"Error loading model: {e}"
    
    summary_lines = []
    
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8", errors="ignore")
            file.seek(0)  # Reset file pointer for potential re-reading
        except Exception as e:
            summary_lines.append(f"**{file.name}**: Error reading file: {e}")
            continue
        
        summary = summarize_content(content, summarizer, tokenizer)
        summary_lines.append(f"**{file.name}**: {summary}")
    
    if not summary_lines:
        return "No uploaded files to summarize."
    return "\n\n".join(summary_lines)