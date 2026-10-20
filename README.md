# CareerCompass

CareerCompass is an AI career coach that analyzes a visitor's own resume PDF and creates a personalized career plan.

## Features

- Resume strengths and skills
- Suggested job roles with evidence from the resume
- Skills gaps and a personalized 30/60/90-day learning roadmap
- Follow-up questions grounded in the uploaded resume
- Downloadable Markdown career plan

## Privacy

The app extracts selectable PDF text in memory and does not save the uploaded file or its text to disk or a database. The text is sent to Google Gemini for the report and follow-up answers. Users should upload only documents they are comfortable processing with Gemini. Scanned PDFs need OCR before upload.

## Run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Set `GEMINI_API_KEY` in an untracked `.env` file at the repository root.
3. Start the app: `streamlit run app.py`

See [DEPLOYMENT.md](./DEPLOYMENT.md) for Streamlit Community Cloud setup. Use a clean GitHub repository for deployment; the existing repository history contains previously tracked personal documents and environment files.
