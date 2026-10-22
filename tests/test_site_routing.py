"""Evidence routing by disease site: the table, scoped FAISS and Knowledge Base searches, and the engine switch."""

import contextlib
import io
import unittest

import faiss
import numpy as np

from config import AWSConfig, Config, RetrieverConfig
from core.case_labels import NOT_SPECIFIED, CaseLabels
from core.models import RetrievalStrategy, RetrievedChunk, SubTask
from core.retriever import RetrieverManager
from core.site_routing import EvidenceScope, scope_for


def bare_retriever():
    """A RetrieverManager without AWS clients or the reranker; tests attach only what they use."""
    retriever = RetrieverManager.__new__(RetrieverManager)
    retriever.config = RetrieverConfig()
    retriever.aws_config = AWSConfig(region="us-west-2", s3_bucket="bucket", knowledge_base_id="kb",
                                     knowledge_base_datasource_id="source")
    retriever.kb_id, retriever.kb_enabled = "kb", True
    retriever._ids_by_category = None
    return retriever


def chunk(category, source_type="review_notes"):
    return RetrievedChunk(faiss_id=1, chunk_id=0, text="Synthetic", path="p", category=category, source_type=source_type)


class TableTests(unittest.TestCase):
    def test_scopes(self):
        self.assertEqual(scope_for(NOT_SPECIFIED), EvidenceScope())
        self.assertEqual(scope_for("Lung & Thoracic"), EvidenceScope(frozenset({"Thorax"}), frozenset({"Lung"})))
        self.assertIn("BrainMets|Oligo|Immuno", scope_for("CNS").review_note_categories)
        self.assertEqual(scope_for("Palliative").review_note_categories, frozenset())   # strict: no notes
        self.assertEqual(scope_for("Benign & HO Prophylaxis").trial_paper_folders, frozenset({"HO Prophylaxis"}))

    def test_allows_checks_each_source_against_its_own_list(self):
        scope = scope_for("Lung & Thoracic")
        self.assertTrue(scope.allows({"source_type": "review_notes", "category": "Thorax"}))
        self.assertFalse(scope.allows({"source_type": "clinical_trial", "category": "Thorax"}))
        self.assertTrue(EvidenceScope().allows({"source_type": "clinical_trial", "category": "Anything"}))


class RetrieverTests(unittest.TestCase):
    def test_dense_search_stays_inside_the_allowed_categories(self):
        retriever = bare_retriever()
        index = faiss.IndexFlatIP(2)
        # IDs 0 and 1 (Thorax) are far from the query; 2 and 3 (Breast) are close
        index.add(np.array([[0, 1], [0.6, 0.8], [1, 0], [0.9, 0.1]], dtype="float32"))
        retriever._load_faiss_index = lambda: index
        retriever._get_embedding = lambda text: np.array([1, 0], dtype="float32")
        retriever._ids_by_category = {"Thorax": np.array([0, 1]), "Breast": np.array([2, 3])}
        retriever._fetch_chunks_from_dynamodb = lambda ids: [{"FAISS_ID": i, "category": "?"} for i in ids]
        retriever.config.similarity_threshold = -1
        found = [c.faiss_id for c in retriever.dense_search("q", top_k=2, categories=frozenset({"Thorax"}))]
        self.assertEqual(found, [1, 0])
        self.assertEqual(retriever.dense_search("q", top_k=2, categories=frozenset()), [])

    def test_category_filter_works_on_an_index_with_stored_ids(self):
        retriever = bare_retriever()
        # The rebuilt index stores FAISS_IDs, which need not match positions
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
        index.add_with_ids(np.array([[0, 1], [0.6, 0.8], [1, 0], [0.9, 0.1]], dtype="float32"),
                           np.array([40, 10, 30, 20], dtype="int64"))
        retriever._load_faiss_index = lambda: index
        retriever._get_embedding = lambda text: np.array([1, 0], dtype="float32")
        retriever._ids_by_category = {"Thorax": np.array([40, 10]), "Breast": np.array([30, 20])}
        retriever._fetch_chunks_from_dynamodb = lambda ids: [{"FAISS_ID": i, "category": "?"} for i in ids]
        retriever.config.similarity_threshold = -1
        found = [c.faiss_id for c in retriever.dense_search("q", top_k=2, categories=frozenset({"Thorax"}))]
        self.assertEqual(found, [10, 40])

    def test_knowledge_base_filter_uses_folder_prefixes(self):
        retriever = bare_retriever()
        calls = []

        class Client:
            def retrieve(self, **kwargs):
                calls.append(kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"])
                return {"retrievalResults": []}

        retriever.bedrock_agent = Client()
        retriever.retrieve_from_knowledge_base("q", folders=frozenset({"Lung"}))
        retriever.retrieve_from_knowledge_base("q", folders=frozenset({"Genitourinary", "Prostate"}))
        self.assertEqual(retriever.retrieve_from_knowledge_base("q", folders=frozenset()), [])
        source, lung = calls[0]["andAll"]
        self.assertEqual(source["equals"]["value"], "source")
        self.assertEqual(lung["startsWith"]["value"], "s3://bucket/reading-lists/Lung/")
        self.assertEqual(len(calls[1]["andAll"][1]["orAll"]), 2)
        self.assertEqual(len(calls), 2)                          # an empty folder set makes no call

    def test_every_strategy_is_checked_against_the_scope(self):
        retriever = bare_retriever()
        retriever.hybrid_search = lambda q, top_k=None: [chunk("Thorax"), chunk("Breast")]
        task = SubTask(task_id="t", description="d", query="q", retrieval_strategy=RetrievalStrategy.HYBRID)
        retriever.execute_task(task, scope_for("Lung & Thoracic"))
        self.assertEqual([c["category"] for c in task.retrieved_chunks], ["Thorax"])


class EngineTests(unittest.TestCase):
    def run_engine(self, site_filter):
        from core.agent import AgenticRAGEngine
        config = Config()
        config.features.site_filter = site_filter
        with contextlib.redirect_stdout(io.StringIO()):
            engine = AgenticRAGEngine(config=config, use_mock=True, enable_judge=False)
            scopes = []
            original = engine.retriever.execute_task
            engine.retriever.execute_task = lambda task, scope=None: (scopes.append(scope), original(task, scope))[1]
            response = engine.query("Synthetic question", CaseLabels(disease_site="Lung & Thoracic"))
        return scopes, response

    def test_labels_scope_retrieval_only_when_switched_on(self):
        scopes, response = self.run_engine(site_filter=True)
        self.assertTrue(scopes and all(s == scope_for("Lung & Thoracic") for s in scopes))
        self.assertEqual(response.routing["review_note_categories"], ["Thorax"])
        scopes, response = self.run_engine(site_filter=False)
        self.assertTrue(all(s == EvidenceScope() for s in scopes))
        self.assertIsNone(response.routing["review_note_categories"])


if __name__ == "__main__":
    unittest.main()
