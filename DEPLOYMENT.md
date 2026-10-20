# Deploy CareerCompass

CareerCompass analyzes a resume PDF selected by each visitor. It does not include a shared resume or career-document index. The app extracts selectable text in memory, generates a career plan with Gemini, and supports follow-up questions grounded in the uploaded resume.

## Privacy behavior

- The app does not write uploaded PDFs or extracted text to disk or a database.
- Extracted text and generated results remain in that visitor's Streamlit session until they clear the results or the session ends.
- The extracted resume text is sent to Google Gemini for analysis and for each follow-up question. Google's service terms and data handling apply.
- Visitors should upload only documents they are comfortable processing with Gemini. The app does not perform OCR; scanned PDFs need searchable text first.

## Publish using a clean GitHub repository

The original repository history contains earlier tracked resumes, a report, a Chroma index, and `src/.env`. Removing those files in a later commit does not erase earlier Git history. Publish this sanitized app from a separate, clean GitHub repository rather than deploying the original repository history.

1. Keep the clean repository limited to `.gitignore`, `README.md`, `DEPLOYMENT.md`, `app.py`, `requirements.txt`, and `src/rag_chain.py`. Do not add `src/.env`, PDFs, `data/chroma_db/`, or `__pycache__/`.
2. In Streamlit Community Cloud, select that new repository, branch `main`, and main file path `app.py`.
3. In app **Settings → Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your Gemini API key"
   ```

4. Deploy and test with a non-sensitive sample resume first. Never put the API key in source code or Git.

If `src/.env` ever contained a real API key, revoke/rotate it; deleting the file does not remove the key from Git history.

## Local run

Install dependencies with `pip install -r requirements.txt`, configure `GEMINI_API_KEY` in an untracked `.env` file at the repository root, and run `streamlit run app.py`.
