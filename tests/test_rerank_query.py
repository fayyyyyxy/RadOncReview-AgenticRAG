"""The rerank query (docs/rag_improvement_plan_v2.md, E2 option (b)): with rerank_site_query on, each topic's
candidates are ranked against the selected disease site and the topic's label instead of the whole vignette and the
label. No AWS."""

import contextlib
import io
import unittest

from config import Config
from core.case_labels import CaseLabels
from core.models import QueryAnalysis, QueryIntent, RetrievalStrategy, SelfCritiqueResult, SubTask, TaskPlan

QUESTION = "Patient vignette:\n65F with cT1N0 rectal adenocarcinoma.\n\nClinical question:\nWhat boost is appropriate?"
SITE = CaseLabels(disease_site="Gastrointestinal")


def make_engine(**features):
    from core.agent import AgenticRAGEngine
    config = Config()
    for name, value in features.items():
        setattr(config.features, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        return AgenticRAGEngine(config=config, use_mock=True, enable_judge=False)


def saved_plan():
    analysis = QueryAnalysis(original_query=QUESTION, intent=QueryIntent.TREATMENT_PROTOCOL, is_complex=True)
    tasks = [SubTask(task_id=f"task_{i}", description=f"[Guidelines] {label}", query=f"saved query {i}",
                     retrieval_strategy=RetrievalStrategy.DENSE, top_k=6)
             for i, label in ((1, "boost dose"), (2, "boost target"))]
    return TaskPlan(plan_id="plan_saved", original_query=QUESTION, analysis=analysis, sub_tasks=tasks,
                    reasoning="Saved reasoning")


class RerankQueryTests(unittest.TestCase):
    """The mock search gives every topic one passage; the queries the reranker is given are recorded."""

    def run_query(self, labels=None, critique=None, replan=None, **features):
        engine = make_engine(**features)
        self.queries = []
        rerank = engine.retriever.rerank
        engine.retriever.rerank = lambda query, chunks, top_k=5: (self.queries.append(query),
                                                                  rerank(query, chunks, top_k))[1]
        engine.self_critic.critique = critique or (lambda *args, **kwargs: SelfCritiqueResult(
            is_complete=True, is_safe=True, confidence_score=0.8))
        if replan:
            engine.task_planner.replan = replan
        with contextlib.redirect_stdout(io.StringIO()):
            engine.query(QUESTION, labels=labels, plan=saved_plan())
        return self.queries

    def test_the_vignette_and_label_by_default(self):
        self.assertEqual(self.run_query(SITE), [f"{QUESTION} boost dose", f"{QUESTION} boost target"])

    def test_the_site_and_label_when_on(self):
        self.assertEqual(self.run_query(SITE, rerank_site_query=True),
                         ["Gastrointestinal boost dose", "Gastrointestinal boost target"])

    def test_no_selected_site_keeps_the_vignette(self):
        self.assertEqual(self.run_query(CaseLabels(), rerank_site_query=True),
                         [f"{QUESTION} boost dose", f"{QUESTION} boost target"])

    def test_a_replanned_topic_without_a_label_ranks_on_the_site(self):
        unsafe = lambda *args, **kwargs: SelfCritiqueResult(is_complete=True, is_safe=False, concerns=["Synthetic"],
                                                            confidence_score=0.8)

        def replan(plan, guidance):
            # Re-planned tasks come back with a blank description in most runs, so their label is empty
            task = SubTask(task_id="task_3", description="", query="re-planned query",
                           retrieval_strategy=RetrievalStrategy.DENSE, top_k=6)
            return TaskPlan(plan_id="plan_re", original_query=QUESTION, analysis=plan.analysis, sub_tasks=[task])

        queries = self.run_query(SITE, critique=unsafe, replan=replan, rerank_site_query=True)
        self.assertEqual(queries, ["Gastrointestinal boost dose", "Gastrointestinal boost target", "Gastrointestinal"])


if __name__ == "__main__":
    unittest.main()
