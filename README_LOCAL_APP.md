## Dama RAG local web app (Ollama)

This folder contains your transcript files (`*.txt`) and a local RAG index (`rag_index/`).
All LLM inference runs locally via **Ollama** -- no API keys or cloud services needed.

### Prerequisites

- **Ollama** installed and running (`ollama serve` or the desktop app)
- A model pulled, e.g. `ollama pull llama3`

### Install (one-time)

Using your Conda Python:

```powershell
cd "C:\Users\ADMIN\Desktop\dama2"
& "C:\Users\ADMIN\miniconda3\python.exe" -m pip install fastapi uvicorn[standard] jinja2 python-multipart requests chromadb sentence-transformers tqdm
```

### Run the app

```powershell
cd "C:\Users\ADMIN\Desktop\dama2"
& "C:\Users\ADMIN\miniconda3\python.exe" -m uvicorn local_app:app --host 127.0.0.1 --port 8000
```

Open in your browser: `http://127.0.0.1:8000`

### Rebuilding the index

Use the **Rebuild Index** button in the UI, or run:

```powershell
cd "C:\Users\ADMIN\Desktop\dama2"
& "C:\Users\ADMIN\miniconda3\python.exe" build_index.py
```

### Notes

- The index builder indexes the per-talk `.txt` files and **skips** `all_transcripts.txt` (too large to embed practically).
- LLM answers are generated locally by Ollama (default model: `llama3`). Toggle the checkbox in the UI to switch between LLM answers and retrieval-only mode.
