# Dama2 -- Local RAG for Anguttara Nikaya Transcripts

A local Retrieval-Augmented Generation (RAG) application for querying Buddhist Dhamma lecture transcripts from the Anguttara Nikaya, as taught by **Bhante Hye Dhammavuddho Mahathera**.

All LLM inference runs locally via [Ollama](https://ollama.com/) -- no API keys or cloud services needed.

## How it works

```
User question
     |
     v
Embedding search (ChromaDB + all-MiniLM-L6-v2)
     |
     v
Top K chunks (with lexical reranking + context expansion)
     |
     v
Local LLM synthesis (Ollama / Qwen 2.5 14B)
     |
     v
Grounded answer with quotes from transcripts
```

## Sample data

This repo includes **1 sample transcript** (`001_*.txt`) for demonstration. The full corpus is 97 lecture transcripts.

To download all transcripts from the [YouTube playlist](https://www.youtube.com/playlist?list=PLD8I9vPmsYXxR_Qt36EbquMkYTOZbXWpM):

```bash
pip install yt-dlp
python download_transcripts.py
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- Pull the LLM model: `ollama pull qwen2.5:14b`

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download all transcripts (or use the 1 sample included):

```bash
python download_transcripts.py
```

3. Build the vector index:

```bash
python build_index.py
```

4. Start the web app:

```bash
python -m uvicorn local_app:app --host 127.0.0.1 --port 8000
```

5. Open http://127.0.0.1:8000 in your browser and ask questions.

## Files

| File | Description |
|------|-------------|
| `local_app.py` | FastAPI web app with UI, retrieval, and LLM synthesis |
| `query_rag.py` | CLI tool for querying the index |
| `build_index.py` | Builds the ChromaDB vector index from transcript files |
| `download_transcripts.py` | Downloads all 97 transcripts from YouTube |
| `start_dama_rag.bat` | Windows launcher script |
| `requirements.txt` | Python dependencies |

## Source

Lecture transcripts are auto-generated from this YouTube playlist:
[Anguttara Nikaya by Bhante Hye Dhammavuddho Mahathera](https://www.youtube.com/playlist?list=PLD8I9vPmsYXxR_Qt36EbquMkYTOZbXWpM)
