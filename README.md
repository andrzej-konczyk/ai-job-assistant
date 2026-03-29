# AI Career Assistant

Streamlit app that ranks job offers against your CV using **semantic similarity** (sentence embeddings), highlights **skill gaps**, and suggests **learning priorities**. Optional **Anthropic Claude** integration adds CV domain classification, richer skill extraction, fit analysis, career-path ideas, and a simple CV Q&A chat.

Job listings come from the **[JustJoin.it](https://justjoin.it)** public offers API when it is reachable; otherwise the app **falls back to a bundled multi-industry mock dataset** so local runs and deploys keep working.

---

## Live app (Streamlit Cloud)

Use the app **without a local install** on Streamlit Community Cloud:

**[https://ai-job-assistant-xqekxfpau3e9fdzfsjvxsi.streamlit.app/](https://ai-job-assistant-xqekxfpau3e9fdzfsjvxsi.streamlit.app/)**

Cold starts (first visit or after sleep) can take a while while the embedding model downloads from Hugging Face.

---

## Features

| Area | Without API key | With Anthropic API key (sidebar) |
|------|-----------------|----------------------------------|
| CV input | PDF upload only (text extracted with [pdfplumber](https://github.com/jsvine/pdfplumber)) | Same |
| Domain / industry | Keyword-based fallback | Claude classification |
| Skills from CV | Keyword list from `matching.SKILLS_VOCAB` | LLM extraction |
| Job matching | Cosine similarity vs job `full_text` embeddings | Same |
| Skill gaps | Embedding similarity per required skill | Same |
| Extra AI | — | Job fit notes, recommendations text, career path, “ask about your CV” |

---

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI, orchestration, styling |
| `config.py` | Paths and defaults (`MODEL_NAME`, thresholds, `jobs.csv` path if used elsewhere) |
| `cv_parser.py` | PDF → plain text |
| `matching.py` | Skill vocabulary, keyword skills, semantic `find_missing_skills`, optional CLI |
| `job_fetcher.py` | `fetch_jobs()` — JustJoin.it → normalized DataFrame, or mock jobs |
| `cv_qa.py` | Claude-powered helpers (classification, chat, career path, etc.) |
| `embeddings.py` | Standalone CLI to rank jobs from a CV PDF |
| `recommendations.py` | Standalone CLI for printed improvement tips |
| `jobs.csv` | Legacy / optional static jobs; **the live app uses `fetch_jobs()`**, not this file by default |
| `tests/` | `pytest` for parser and matching helpers |

---

## Requirements

- **Python 3.10+** (3.11+ recommended)
- Enough disk/RAM for the first run of **`sentence-transformers`** (downloads `all-MiniLM-L6-v2` from Hugging Face)

---

## Install

```bash
cd ai-job-assistant
python -m venv .venv
```

**Windows (PowerShell):** `.venv\Scripts\Activate.ps1`  
**macOS / Linux:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

---

## Run the app locally

**Recommended:**

```bash
streamlit run app.py
```

**Shortcut (forwards to Streamlit):**

```bash
python app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`). Upload a **text-based PDF** CV. Scanned image-only PDFs may not extract text.

> **Hosted:** use the [Streamlit Cloud deployment](https://ai-job-assistant-xqekxfpau3e9fdzfsjvxsi.streamlit.app/) instead of running locally (see *Live app* above).

---

## Optional: Anthropic API key

In the sidebar, paste an [Anthropic API key](https://docs.anthropic.com/en/api/getting-started) to enable Claude features. The key is used only in the browser session for that run (not written to the repo).

For **Streamlit Community Cloud**, you can store the key under **App settings → Secrets** and wire it in code if you add `st.secrets` support; the current UI expects manual entry in the sidebar.

---

## Command-line tools

All CV paths below are **explicit** — there is no hardcoded `cv.pdf`.

```bash
python cv_parser.py path/to/cv.pdf
python matching.py path/to/cv.pdf [--jobs jobs.csv] [--top-n 5]
python embeddings.py path/to/cv.pdf [--jobs jobs.csv] [--top-n 5]
python recommendations.py path/to/cv.pdf [--jobs jobs.csv] [--top-n 5]
```

Note: `matching.py` / `embeddings.py` / `recommendations.py` default to **`config.JOBS_CSV`** when `--jobs` is omitted; the Streamlit app itself loads data via **`job_fetcher.fetch_jobs()`**, not that CSV.

---

## Tests

```bash
pytest tests/
```

---

## Deploy (Streamlit Cloud)

**Current public instance:** [ai-job-assistant-xqekxfpau3e9fdzfsjvxsi.streamlit.app](https://ai-job-assistant-xqekxfpau3e9fdzfsjvxsi.streamlit.app/).

To deploy your own fork or a new app:

1. Set the entrypoint to **`app.py`**.
2. Use **`requirements.txt`** for dependencies.
3. Expect **cold starts**: the embedding model downloads on first run (free tier can take several minutes).
4. Outbound **HTTPS** must work for JustJoin.it and (optionally) Anthropic; if JustJoin fails, the app still runs on the **mock** job set.

---

## Limitations

- Job API and model download depend on **external services**; failures are handled with fallbacks or error messages where implemented.
- **GDPR / terms**: when using live job data, respect [JustJoin.it](https://justjoin.it) and your jurisdiction’s rules for automated processing and storage.
- Semantic matching is **heuristic**; always treat suggestions as advisory, not hiring decisions.

---

## License

Add your preferred license file if you publish the repository; this README does not impose one.
