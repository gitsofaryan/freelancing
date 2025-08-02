import os
from transformers import pipeline

# Supported file extensions
FILE_EXTS = {".py", ".sql", ".yml", ".yaml", ".xml", ".conf", ".ini", ".txt"}

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

def chunk_text(text, max_tokens=500):
    # Simple chunking for long files
    lines = text.splitlines()
    chunks = []
    chunk = []
    token_count = 0
    for line in lines:
        token_count += len(line.split())
        if token_count > max_tokens:
            chunks.append("\n".join(chunk))
            chunk = []
            token_count = 0
        chunk.append(line)
    if chunk:
        chunks.append("\n".join(chunk))
    return chunks

def summarize_file(file_path, summarizer):
    content = read_file(file_path)
    if not content.strip():
        return "File is empty or unreadable."
    # Chunk if too long
    chunks = chunk_text(content, max_tokens=500)
    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=60, min_length=15, do_sample=False)
        summaries.append(result[0]['summary_text'])
        if len(summaries) >= 2:  # Limit to 2 summaries per file
            break
    return " ".join(summaries)

def summarize_files(directory):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    files = get_files_in_directory(directory)
    summary_lines = []
    for file_path in files:
        file_name = os.path.relpath(file_path, directory)
        summary = summarize_file(file_path, summarizer)
        summary_lines.append(f"**{file_name}**: {summary}")
    if not summary_lines:
        return "No supported files found in the directory."
    return "\n\n".join(summary_lines)
