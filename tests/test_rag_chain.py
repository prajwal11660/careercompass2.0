import hashlib
import re
import tempfile
import unittest
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.rag_chain import (
    analyze_resume,
    answer_resume_question,
    build_career_knowledge_store,
    build_resume_store,
    format_references,
    retrieve_context,
)


class TinyEmbeddings:
    def _embed(self, text):
        vector = [0.0] * 64
        for token in re.findall(r"\w+", text.lower()):
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big") % len(vector)
            vector[index] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


class RagChainTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.knowledge_dir = Path(self.temp_dir.name)
        (self.knowledge_dir / "analytics.md").write_text(
            "# Data analytics\n\n"
            "Reference: [National Career Service](https://www.ncs.gov.in/)\n\n"
            "SQL, dashboards, data quality, and communication are useful data analyst skills.",
            encoding="utf-8",
        )
        self.embeddings = TinyEmbeddings()
        self.knowledge_store = build_career_knowledge_store(
            self.embeddings,
            self.knowledge_dir,
        )
        self.resume_text = (
            "[Page 1]\nBuilt SQL dashboards and analyzed customer data for a class project."
        )
        self.resume_store = build_resume_store(self.resume_text, self.embeddings)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resume_and_public_knowledge_indexes_are_separate(self):
        resume_matches = retrieve_context(self.resume_store, "SQL dashboards", k=1)
        knowledge_matches = retrieve_context(
            self.knowledge_store,
            "SQL data analyst dashboards",
            k=1,
        )

        self.assertEqual(resume_matches[0].metadata["source"], "Uploaded resume")
        self.assertEqual(resume_matches[0].metadata["page"], 1)
        self.assertEqual(knowledge_matches[0].metadata["source"], "analytics.md")
        self.assertNotIn("customer data for a class project", knowledge_matches[0].page_content)

    def test_retrieved_knowledge_has_clickable_reference(self):
        matches = retrieve_context(self.knowledge_store, "SQL analytics", k=1)

        self.assertIn(
            "[National Career Service](https://www.ncs.gov.in/)",
            format_references(matches),
        )

    def test_initial_report_uses_rag_and_appends_sources(self):
        llm = FakeListChatModel(responses=["Career plan"])

        result = analyze_resume(
            self.resume_text,
            self.resume_store,
            self.knowledge_store,
            llm,
        )

        self.assertIn("Career plan", result)
        self.assertIn("Retrieved career resources", result)
        self.assertIn("https://www.ncs.gov.in/", result)

    def test_follow_up_retrieves_both_stores_and_history(self):
        llm = FakeListChatModel(responses=["Use SQL project evidence to target analyst roles."])

        result = answer_resume_question(
            self.resume_store,
            self.knowledge_store,
            "How can I target data analyst roles?",
            llm,
            chat_history="user: What skills do I show?",
        )

        self.assertIn("analyst roles", result)
        self.assertIn("https://www.ncs.gov.in/", result)


if __name__ == "__main__":
    unittest.main()
