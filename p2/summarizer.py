import os
from collections import defaultdict
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
            if len(summaries) >= 2:  # Keep it concise
                break
        except Exception as e:
            continue
    
    if not summaries:
        return "Could not generate summary for this file."
    
    return " ".join(summaries)

def create_consolidated_summary(file_counts, individual_summaries):
    # Create overview
    overview_parts = []
    for ext, count in file_counts.items():
        if ext == ".py":
            overview_parts.append(f"{count} Python scripts")
        elif ext in [".yml", ".yaml"]:
            overview_parts.append(f"{count} YAML files")
        elif ext == ".sql":
            overview_parts.append(f"{count} SQL files")
        else:
            overview_parts.append(f"{count} {ext[1:].upper()} files")
    
    overview = f"This directory contains {', '.join(overview_parts)}."
    
    # Combine with individual summaries
    return f"{overview}\n\n{individual_summaries}"

def summarize_files(directory):
    try:
        summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        return f"Error loading model: {e}"
    
    files, file_counts = get_files_in_directory(directory)
    summary_lines = []
    
    for file_path in files:
        file_name = os.path.relpath(file_path, directory)
        summary = summarize_content(read_file(file_path), summarizer, tokenizer)
        
        # Add context based on file type
        ext = os.path.splitext(file_name)[-1].lower()
        if ext == ".sql":
            summary_lines.append(f"**{file_name}** (SQL): {summary}")
        elif ext == ".py":
            summary_lines.append(f"**{file_name}** (Python): {summary}")
        elif ext in [".yml", ".yaml"]:
            summary_lines.append(f"**{file_name}** (YAML): {summary}")
        else:
            summary_lines.append(f"**{file_name}**: {summary}")
    
    if not summary_lines:
        return "No supported files found in the directory."
    
    individual_summaries = "\n\n".join(summary_lines)
    return create_consolidated_summary(file_counts, individual_summaries)
