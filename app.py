import os
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
from src.rag_chain import analyze_resume, answer_resume_question, extract_resume_text, get_llm

load_dotenv()

st.set_page_config(
    page_title="CareerCompass AI",
    page_icon="🚀",
    layout="centered"
)

st.title("CareerCompass — Your AI Career Coach")
st.subheader("Get personalized career guidance from your resume")
st.warning(
    "Privacy: your PDF is processed in this app session and is not saved to disk or a "
    "database by this app. Its extracted text is sent to Google Gemini to generate "
    "answers. Only upload a document you are comfortable sharing with that service."
)

if not os.getenv("GEMINI_API_KEY"):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, StreamlitSecretNotFoundError):
        st.error(
            "Gemini is not configured. Add GEMINI_API_KEY in Streamlit app Secrets, "
            "or set it in your local .env file."
        )
        st.stop()
    if not api_key:
        st.error("GEMINI_API_KEY is empty. Add a valid key in Streamlit app Secrets.")
        st.stop()
    os.environ["GEMINI_API_KEY"] = api_key


@st.cache_resource
def load_llm():
    return get_llm()

llm = load_llm()
st.session_state.setdefault("upload_version", 0)

with st.form(key=f"resume_form_{st.session_state.upload_version}"):
    uploaded_file = st.file_uploader(
        "Upload your resume as a text-searchable PDF (max 10 MB, 20 pages)",
        type=["pdf"],
        max_upload_size=10,
        key=f"resume_upload_{st.session_state.upload_version}",
    )
    submitted = st.form_submit_button("Analyze my resume")

if submitted:
    if uploaded_file is None:
        st.error("Choose a resume PDF first.")
    else:
        try:
            resume_text = extract_resume_text(uploaded_file.getvalue())
        except ValueError as exc:
            st.error(str(exc))
        else:
            with st.spinner("Reviewing your resume and preparing a career plan..."):
                analysis = analyze_resume(resume_text, llm)
            st.session_state["resume_text"] = resume_text
            st.session_state["resume_analysis"] = analysis
            st.session_state["messages"] = []

if "resume_analysis" in st.session_state:
    st.markdown(st.session_state["resume_analysis"])
    st.download_button(
        "Download your career plan",
        data=st.session_state["resume_analysis"],
        file_name="careercompass-career-plan.md",
        mime="text/markdown",
    )

    st.subheader("Ask a follow-up question")
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask about skills, roles, or your roadmap..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_resume_question(
                    st.session_state["resume_text"], question, llm
                )
            st.markdown(answer)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
            })

    if st.button("Clear resume and results"):
        for key in ("resume_text", "resume_analysis", "messages"):
            st.session_state.pop(key, None)
        st.session_state["upload_version"] += 1
        st.rerun()