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

## Cloud Run deploy (topic search + Vertex chat)

This repo also contains the AN topic-search app (`topic_search_server.py`) which can be deployed to Google Cloud Run.

### CI/CD (GitHub Actions → Cloud Run)

- **DAMA:** `.github/workflows/deploy-cloudrun-dama.yml` → service **`dama`** (root `Dockerfile`, `topic_search_server.py`).
- **AN1 RAG:** `.github/workflows/deploy-cloudrun-an1.yml` → service **`dama-an1`** (`Dockerfile.an1`, bundled `an1.json` / `an2.json`).
- On push to `main`, **only the workflow whose paths changed** runs (plus manual **Run workflow** on either file). Both can run in parallel if both path sets match in one push.
- Deploys with **`--allow-unauthenticated`** so **normal browsers** can reach the service (Cloud Run IAM does **not** use “signed into Gmail” as proof — that’s why `roles/run.invoker` on your user still returns **403** in Chrome).

**App-level “private link” (recommended for phone browsers):**

1. Add a GitHub Actions secret **`DAMA_ACCESS_KEY`** (long random string; **no commas**).
2. Deploy (push to `main` or run the workflow). The workflow passes `DAMA_ACCESS_KEY` into the container.
3. Open once in the same browser you use daily:

   `https://YOUR-SERVICE-URL.run.app/?access_key=YOUR_SECRET`

   The app sets an **HttpOnly cookie** (30 days) so `/api/*` works without putting the key on every request.

4. Optional: remove IAM user bindings you added for testing; **`allUsers`** invoker is expected while using `DAMA_ACCESS_KEY`.

If `DAMA_ACCESS_KEY` is **unset**, the app stays **fully public** (same as no gate).

**Desktop-only access without changing deploy (IAM-only Cloud Run):**

```bash
gcloud run services proxy dama --region=us-central1 --project=dama-492316
```

You must set up Workload Identity Federation (recommended) and add GitHub repo secrets:
- `GCP_WIF_PROVIDER`
- `GCP_WIF_SERVICE_ACCOUNT`

### Private data in GCS (recommended)

The Cloud Run service can load `an*.json` from a **private** GCS bucket/prefix via:
- `DAMA_DATA_GCS_URI=gs://<bucket>/<prefix>/`

At startup, the app downloads matching `an*.json` into `/tmp` and uses them for topic search.

### Vertex AI chat

Enable Vertex chat with:
- `DAMA_USE_VERTEX=1`
- `DAMA_VERTEX_MODEL=gemini-2.5-flash` (default)
- `DAMA_MAX_OUTPUT_TOKENS=384` (optional cap)

### Budget kill switch (disable chat only)

Set `DAMA_DISABLE_CHAT=1` on the Cloud Run service to disable `/api/chat` while keeping topic search available.
There’s a Pub/Sub-triggered Cloud Function stub under `gcp/budget_kill_chat/` that can be wired to a $2.50 billing budget.

### AN1 app (sutta + commentary) on Cloud Run (Vertex)

- **Data in git:** `processed scipts2/an1.json` is **not** ignored (see `.gitignore` exception) so **`Dockerfile.an1`** and CI always have the sutta JSON.
- **Deploy workflow:** `.github/workflows/deploy-cloudrun-an1.yml` builds and deploys the **application image** (`Dockerfile.an1` → **`dama-an1`**). **Embeddings / Vertex shards** are built separately by `.github/workflows/build-vertex-corpus.yml` and uploaded to GCS; Cloud Run reads them via `AN1_VERTEX_MANIFEST_GCS_URI` / `AN1_VERTEX_BUNDLE_GCS_URI`.
- **CLI-only deploy (no local Docker):** `gcloud builds submit --config=cloudbuild.an1.yaml --project=dama-492316 .` then if needed `gcloud run services add-iam-policy-binding dama-an1 --region=us-central1 --member=allUsers --role=roles/run.invoker --project=dama-492316`.
- **Runtime env:** `AN1_USE_VERTEX=1`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`, `AN1_VERTEX_BUNDLE_GCS_URI`, optional `DAMA_VERTEX_MODEL`, `AN1_VERTEX_EMBEDDING_MODEL`, `DAMA_MAX_OUTPUT_TOKENS`.
- **IAM:** See comments in `.github/workflows/deploy-cloudrun-an1.yml` (Vertex + GCS; deploy is **unauthenticated** unless you lock down Cloud Run separately).
