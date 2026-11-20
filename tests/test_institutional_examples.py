"""Example/evidence separation and leakage checks with synthetic cases; no AWS."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from config import Config
from core.agent import AgenticRAGEngine
from core.case_labels import CaseLabels
from core.institutional_examples import CaseContext, LocalInstitutionalExamples
from core.models import QueryAnalysis, QueryIntent, SelfCritiqueResult, TaskPlan
from core.past_cases import LABEL, NumpyPastCaseIndex, search_text
from core.synthesizer import ResponseSynthesizer


MODEL = "synthetic-embedding-model"
LABELS = CaseLabels(disease_site="Breast", setting_group="Adjuvant", prior_rt="None", encounter_type="New start")
CONTEXT = CaseContext("Synthetic vignette SELF", "Synthetic question", "SELF", "PATIENT_SELF")
TYPED = "Typed web question about adjuvant breast radiotherapy"
EVIDENCE = {"text": "Synthetic evidence statement.", "path": "clinical-reference",
            "source_type": "review_notes", "faiss_id": 1, "similarity_score": 0.8}


def write_index(root):
    examples, records = [], []
    for case_id, patient, site in (("SELF", "PATIENT_SELF", "Breast"), ("TWIN", "PATIENT_SELF", "Breast"),
                                   ("GOOD", "PATIENT_OTHER", "Breast"), ("ALSO", "PATIENT_THIRD", "Breast"),
                                   ("OTHER_SITE", "PATIENT_FOURTH", "CNS")):
        example = {
            "case_id": case_id, "vignette": f"Synthetic vignette {case_id}",
            "clinical_question": "Synthetic question", "attending_plan": f"PLAN_{case_id}",
            "patient_linkage": {"patient_id": patient},
            "metadata": {"disease_site": site, "setting": "Adjuvant", "prior_rt": "None", "encounter_type": "New start"},
            "governance": {"split": "Development", "worked_example_eligible": "Yes"},
        }
        examples.append(example)
        records.append({
            **{key: example[key] for key in ("case_id", "vignette", "clinical_question", "attending_plan")},
            "patient_id": patient, "disease_site": site, "setting_group": "Adjuvant",
            "prior_rt_group": "No prior RT", "encounter_type": "New start", "label": LABEL,
            "search_sha256": hashlib.sha256(search_text(example["vignette"], example["clinical_question"]).encode()).hexdigest(),
        })
    source = root / "development_examples.jsonl"
    source.write_text("".join(json.dumps(example) + "\n" for example in examples))
    index = NumpyPastCaseIndex(3)
    index.add(records, np.array([[1, 0, 0]] * len(records)))
    folder = root / "past_case_index"
    index.save(folder)
    manifest = {"embedding_model": MODEL, "dimension": 3, "cases": len(records),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "version": index.version()}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


class FakeWriter:
    def generate(self, prompt, system_prompt=None):
        return "Synthetic evidence statement [1]."


class IndexAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = write_index(Path(self.temp.name))
        self.provider = LocalInstitutionalExamples(self.folder, MODEL, 3)

    def test_excludes_self_same_patient_and_other_site_reusing_known_vector(self):
        embed = Mock(return_value=np.array([1, 0, 0]))
        examples = self.provider.select(CONTEXT.query(), LABELS, embed, CONTEXT)
        self.assertEqual([e["case_id"] for e in examples], ["GOOD", "ALSO"])
        embed.assert_not_called()
        self.assertTrue(all("patient_id" not in e and e["label"] == LABEL for e in examples))

    def test_unindexed_case_embeds_only_vignette_and_question(self):
        embed = Mock(return_value=np.array([1, 0, 0]))
        query = CaseContext("Unseen synthetic vignette", "Unseen question", "UNSEEN_CASE", "UNSEEN_PATIENT")
        self.provider.select(query.query(), LABELS, embed, query)
        embed.assert_called_once_with(search_text(query.vignette, query.clinical_question))

    def test_query_without_case_context_is_embedded_as_typed_and_excludes_nothing(self):
        embed = Mock(return_value=np.array([1, 0, 0]))
        examples = self.provider.select(TYPED, LABELS, embed)
        embed.assert_called_once_with(TYPED)
        # Equal similarities keep index order, so the first two Breast cases come back: nothing was excluded
        self.assertEqual([e["case_id"] for e in examples], ["SELF", "TWIN"])

    def test_wrong_embedding_model_and_stale_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "embedding model"):
            LocalInstitutionalExamples(self.folder, "other-model", 3)
        source = self.folder.parent / "development_examples.jsonl"
        source.write_text(source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "development export"):
            LocalInstitutionalExamples(self.folder, MODEL, 3)

    def test_index_from_another_workbook_corpus_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different workbook"):
            LocalInstitutionalExamples(self.folder, MODEL, 3, expected_source_sha256="different-source")

    def test_locked_test_record_is_rejected_even_with_matching_source_hash(self):
        source = self.folder.parent / "development_examples.jsonl"
        source.write_text(source.read_text().replace('"split": "Development"', '"split": "Locked test"', 1))
        manifest_path = self.folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "eligible Development"):
            LocalInstitutionalExamples(self.folder, MODEL, 3)

    def test_invalid_query_linkage_is_rejected_before_embedding(self):
        embed = Mock()
        invalid = CaseContext(CONTEXT.vignette, CONTEXT.clinical_question, CONTEXT.case_id, "WRONG_PATIENT")
        with self.assertRaisesRegex(ValueError, "linkage"):
            self.provider.select(invalid.query(), LABELS, embed, invalid)
        embed.assert_not_called()

    def test_missing_site_does_not_search_other_sites(self):
        embed = Mock()
        self.assertEqual(self.provider.select(CONTEXT.query(), CaseLabels(), embed, CONTEXT), [])
        self.assertEqual(self.provider.select(TYPED, CaseLabels(), embed), [])
        embed.assert_not_called()


class SynthesisBoundaryTests(unittest.TestCase):
    def test_examples_are_separate_from_numbered_passages_in_both_paths(self):
        examples = [{"case_id": "EXAMPLE_STUDY_ID", "vignette": "Historical case",
                     "clinical_question": "Historical question", "attending_plan": "Historical plan"}]
        analysis = QueryAnalysis("Synthetic question", QueryIntent.GENERAL)
        for multi in (False, True):
            with self.subTest(multi=multi):
                synth = ResponseSynthesizer(FakeWriter(), Config())
                if multi:
                    _, citations, _, passages = synth.synthesize_multi_topic(
                        "Synthetic question", analysis, {"Topic": [EVIDENCE]}, institutional_examples=examples)
                else:
                    _, citations, _, passages = synth.synthesize(
                        "Synthetic question", analysis, None, [EVIDENCE], institutional_examples=examples)
                request = synth.requests[0]
                self.assertIn("Historical plan", request["prompt"])
                self.assertNotIn("EXAMPLE_STUDY_ID", request["prompt"])
                self.assertIn("not clinical evidence", request["system_prompt"])
                self.assertEqual([c.source_path for c in citations], ["clinical-reference"])
                self.assertEqual([p["text"] for p in passages], [EVIDENCE["text"]])

    def test_no_examples_keep_the_evidence_only_prompts(self):
        analysis = QueryAnalysis("Synthetic question", QueryIntent.GENERAL)
        synth = ResponseSynthesizer(FakeWriter(), Config())
        synth.synthesize("Synthetic question", analysis, None, [EVIDENCE])
        synth.synthesize("Synthetic question", analysis, None, [EVIDENCE], institutional_examples=[])
        self.assertEqual(synth.requests[0], synth.requests[1])
        self.assertEqual(synth.requests[0]["system_prompt"], ResponseSynthesizer.SYSTEM_PROMPT)

    def test_examples_alone_cannot_generate_an_answer(self):
        writer = Mock()
        synth = ResponseSynthesizer(writer, Config())
        analysis = QueryAnalysis("Synthetic question", QueryIntent.GENERAL)
        for multi in (False, True):
            if multi:
                result = synth.synthesize_multi_topic("Synthetic question", analysis, {"Topic": []},
                                                     institutional_examples=[{"attending_plan": "Not evidence"}])
            else:
                result = synth.synthesize("Synthetic question", analysis, None, [],
                                          institutional_examples=[{"attending_plan": "Not evidence"}])
            self.assertIn("could not find relevant information", result[0])
            self.assertEqual(result[1:], ([], 0.0, []))
        writer.generate.assert_not_called()


class EngineConnectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = write_index(Path(temp.name))
        self.provider = LocalInstitutionalExamples(self.folder, MODEL, 3)

    def engine(self, enabled=True, complex_query=False):
        config = Config()
        config.institutional_examples.enabled = enabled
        config.agent.enable_self_critique = False
        engine = AgenticRAGEngine(config, use_mock=True, enable_judge=False, example_provider=self.provider)
        analysis = QueryAnalysis(CONTEXT.query(), QueryIntent.GENERAL, is_complex=complex_query, confidence=1)
        engine.query_analyzer.analyze = Mock(return_value=analysis)
        engine.query_analyzer.analyze_with_llm = Mock(return_value=analysis)
        engine.task_planner.plan = Mock(return_value=TaskPlan("first", CONTEXT.query(), analysis, []))
        engine._collect_all_chunks = Mock(return_value=[EVIDENCE])
        engine._collect_multi_topic_chunks = Mock(return_value={"Topic": [EVIDENCE]})
        engine.retriever._get_embedding = Mock(return_value=np.array([1, 0, 0]))
        engine.synthesizer.llm = FakeWriter()
        check_report = {"removed": 0, "citations": 1, "unchecked": 0, "error": None}
        engine.citation_checker.check = Mock(return_value=("Synthetic evidence statement [1].", check_report))
        return engine

    def test_examples_reach_writer_and_citation_check_stays_evidence_only(self):
        for complex_query in (False, True):
            with self.subTest(complex_query=complex_query), contextlib.redirect_stdout(io.StringIO()):
                engine = self.engine(complex_query=complex_query)
                response = engine.query(CONTEXT.query(), LABELS, case_context=CONTEXT)
                self.assertEqual([e["case_id"] for e in response.institutional_examples["examples"]], ["GOOD", "ALSO"])
                self.assertIn("PLAN_GOOD", response.synthesis_requests[0]["prompt"])
                self.assertNotIn("PLAN_SELF", response.synthesis_requests[0]["prompt"])
                self.assertNotIn("PATIENT_SELF", response.synthesis_requests[0]["prompt"])
                checked = engine.citation_checker.check.call_args.args[1]
                self.assertEqual([p["text"] for p in checked], [EVIDENCE["text"]])

    def test_web_query_with_a_site_embeds_the_typed_query_and_keeps_plans_out_of_metadata(self):
        with contextlib.redirect_stdout(io.StringIO()):
            response = self.engine().query(TYPED, LABELS)
        self.assertEqual(response.institutional_examples["status"], "selected")
        self.assertIn("PLAN_SELF", response.synthesis_requests[0]["prompt"])
        self.assertEqual(response.metadata["institutional_examples"], {"status": "selected", "count": 2})
        self.assertNotIn("PLAN_", json.dumps(response.metadata))

    def test_web_query_without_a_site_searches_no_examples(self):
        with contextlib.redirect_stdout(io.StringIO()):
            engine = self.engine()
            response = engine.query(TYPED, CaseLabels())
        self.assertEqual(response.institutional_examples["status"], "no_disease_site")
        self.assertNotIn("Institutional examples", response.synthesis_requests[0]["prompt"])
        engine.retriever._get_embedding.assert_not_called()

    def test_off_switch_does_not_search_examples(self):
        with contextlib.redirect_stdout(io.StringIO()):
            engine = self.engine(enabled=False)
            with patch.object(self.provider, "select", side_effect=AssertionError("must not search")):
                response = engine.query(CONTEXT.query(), LABELS)
        self.assertEqual(response.institutional_examples, {"status": "off"})
        self.assertNotIn("PLAN_GOOD", response.synthesis_requests[0]["prompt"])
        engine.retriever._get_embedding.assert_not_called()

    def test_query_context_must_match_before_analysis_or_embedding(self):
        engine = self.engine()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "must match the query"):
            engine.query("Different question", LABELS, case_context=CONTEXT)
        engine.query_analyzer.analyze.assert_not_called()
        engine.retriever._get_embedding.assert_not_called()

    def test_retry_keeps_examples_and_the_feedback_and_evidence_merge(self):
        for complex_query in (False, True):
            with self.subTest(complex_query=complex_query), contextlib.redirect_stdout(io.StringIO()):
                engine = self.engine(complex_query=complex_query)
                engine.config.agent.enable_self_critique = True
                engine.self_critic.critique = Mock(return_value=SelfCritiqueResult(is_complete=False, is_safe=True))
                engine.self_critic.should_retry = Mock(return_value=True)
                engine.self_critic.get_improvement_guidance = Mock(return_value="Address the requested item")
                first = engine.task_planner.plan.return_value
                engine.task_planner.replan = Mock(return_value=TaskPlan("retry", first.original_query, first.analysis, []))
                added = {**EVIDENCE, "faiss_id": 2, "text": "Additional evidence", "path": "second-reference"}
                engine._collect_all_chunks.side_effect = [[EVIDENCE], [added]]
                engine._collect_multi_topic_chunks.side_effect = [{"Topic": [EVIDENCE]}, {"Topic": [added]}]
                response = engine.query(CONTEXT.query(), LABELS, case_context=CONTEXT)
                self.assertEqual(len(response.synthesis_requests), 2)
                self.assertIn("Address the requested item", response.synthesis_requests[1]["prompt"])
                self.assertTrue(all("PLAN_GOOD" in r["prompt"] for r in response.synthesis_requests))
                self.assertEqual([p["text"] for p in response.context_passages], [EVIDENCE["text"], "Additional evidence"])
                engine.retriever._get_embedding.assert_not_called()


class MissingIndexTests(unittest.TestCase):
    """The web app runs on without an index; the runner, which sets require_index, stops."""

    def config(self, require_index):
        config = Config()
        config.institutional_examples.index_path = str(Path(tempfile.gettempdir()) / "no-such-past-case-index")
        config.institutional_examples.require_index = require_index
        return config

    @patch("core.agent.RetrieverManager")
    @patch("core.agent.create_llm_client")
    def test_web_app_starts_and_reports_the_index_unavailable(self, _llm, _retriever):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            engine = AgenticRAGEngine(self.config(require_index=False), enable_judge=False)
        self.assertIn("without institutional examples", log.getvalue())
        examples, report = engine._select_examples(TYPED, LABELS, None, evidence_count=1)
        self.assertEqual(examples, [])
        self.assertEqual(report["status"], "index_unavailable")
        self.assertIn("missing", report["reason"])

    @patch("core.agent.RetrieverManager")
    @patch("core.agent.create_llm_client")
    def test_runner_stops_before_any_client_is_made(self, llm, retriever):
        with self.assertRaisesRegex(ValueError, "missing"):
            AgenticRAGEngine(self.config(require_index=True), enable_judge=False)
        llm.assert_not_called()
        retriever.assert_not_called()


if __name__ == "__main__":
    unittest.main()
