# CareerCompass

CareerCompass is a hybrid retrieval-augmented AI career coach. It analyzes each visitor's resume in an isolated, temporary vector store and retrieves relevant advice from a separate, shared career-knowledge vector store.

## Features

- Resume strengths and skills
- Suggested job roles with evidence from the resume
- Skills gaps and a personalized 30/60/90-day learning roadmap
- Follow-up questions grounded in the uploaded resume
- Relevant career guidance retrieved from a project-authored India/global knowledge base
- Clickable official career and qualification resources used as references
- Downloadable Markdown career plan

## How RAG works

1. Markdown career resources in `data/knowledge/` are split into overlapping chunks and embedded with `sentence-transformers/all-MiniLM-L6-v2`.
2. Those public resources are indexed in a shared, in-memory LangChain `InMemoryVectorStore`.
3. Each uploaded resume is split and embedded into a separate in-memory vector store kept only in that visitor's Streamlit session.
4. For the initial report, the full extracted resume and retrieved resume evidence are sent to Gemini alongside the most relevant career-knowledge chunks. Follow-up answers retrieve relevant chunks from both stores.
5. Retrieved official links are included as references. The corpus is project-authored guidance, not an official job or qualification standard.

The vector stores exist only in app memory. The shared corpus is safe to cache between sessions; resume stores are never cached globally or shared between visitors. App restarts clear all in-memory indexes.

## Privacy

The app extracts selectable PDF text in memory and does not save the uploaded file or its text to disk or a database. Resume chunks are kept in the current Streamlit session's private in-memory vector store. The text and retrieved excerpts are sent to Google Gemini for the report and follow-up answers. Users should upload only documents they are comfortable processing with Gemini. Scanned PDFs need OCR before upload.

## Run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Set `GEMINI_API_KEY` in an untracked `.env` file at the repository root.
3. Start the app: `streamlit run app.py`

See [DEPLOYMENT.md](./DEPLOYMENT.md) for Streamlit Community Cloud setup. Use a clean GitHub repository for deployment; the existing repository history contains previously tracked personal documents and environment files.
