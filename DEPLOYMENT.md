# Deploy CareerCompass

CareerCompass analyzes a resume PDF selected by each visitor and uses hybrid RAG: an in-memory vector store for that visitor's resume plus a shared in-memory vector store built from project-authored career guidance in `data/knowledge/`.

## Privacy behavior

- The app does not write uploaded PDFs, extracted text, or resume embeddings to disk or a database.
- Resume text, its private in-memory vector store, and generated results remain in that visitor's Streamlit session until they clear the results or the session ends.
- The full resume is sent to Google Gemini for the initial report; retrieved resume and career chunks are sent for follow-up answers. Google's service terms and data handling apply.
- Visitors should upload only documents they are comfortable processing with Gemini. The app does not perform OCR; scanned PDFs need searchable text first.
- The shared career-resource index contains only project-authored guidance and official reference links. It is safe to cache across sessions. Both vector stores are in memory and are cleared when the app process restarts.

## Publish using a clean GitHub repository

The original repository history contains earlier tracked resumes, a report, a Chroma index, and `src/.env`. Removing those files in a later commit does not erase earlier Git history. Publish this sanitized app from a separate, clean GitHub repository rather than deploying the original repository history.

1. Keep the clean repository limited to app source, requirements, docs, `.gitignore`, and the reviewed Markdown files in `data/knowledge/`. Do not add `src/.env`, uploaded resumes, `data/chroma_db/`, or `__pycache__/`.
2. In Streamlit Community Cloud, select that new repository, branch `main`, and main file path `app.py`.
3. In app **Settings → Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your Gemini API key"
   ```

4. Deploy and test with a non-sensitive sample resume first. Never put the API key in source code or Git.

If `src/.env` ever contained a real API key, revoke/rotate it; deleting the file does not remove the key from Git history.

## Local run

Install dependencies with `pip install -r requirements.txt`, configure `GEMINI_API_KEY` in an untracked `.env` file at the repository root, and run `streamlit run app.py`. First startup downloads the embedding model and builds the shared career-knowledge index in memory; subsequent reruns reuse the cached public index.
