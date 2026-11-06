"""Critic edits (docs/rag_improvement_plan_v2.md, E4): the critique reads the passages and returns sentence edits,
which code applies; a gap no passage covers is searched for, and new text goes through the critique once more. The
answer is rewritten only when its main recommendation is wrong. No AWS."""

import contextlib
import io
import unittest

from config import Config
from core.critic_edits import GapFiller, apply_edits
from core.models import (QueryAnalysis, QueryIntent, RetrievalStrategy, SelfCritiqueResult, SubTask, TaskPlan,
                         TaskStatus)
from core.synthesizer import SelfCritic

QUESTION = "Synthetic question"
DRAFT = ("- **Boost dose:** **10.8 Gy in 6 fractions** [3].\n"
         "- MRC CR07 used 25 Gy in 5 fractions and showed local control [5], [6].\n"
         "- Follow-up imaging is advised.")
GAP = {"item": "OAR constraints", "query": "D2cc constraints"}


def edit(sentence, text, action="replace", kind="harm"):
    return {"kind": kind, "action": action, "sentence": sentence, "text": text}


class ApplyEditsTests(unittest.TestCase):
    def test_only_the_named_sentence_changes(self):
        answer, report = apply_edits(DRAFT, [edit("Follow-up imaging is advised.",
                                                  "Follow-up is not covered by the provided evidence.", kind="cite")])
        self.assertEqual(answer, DRAFT.replace("Follow-up imaging is advised.",
                                               "Follow-up is not covered by the provided evidence."))
        self.assertEqual(len(report["applied"]), 1)

    def test_a_quote_without_bold_citations_or_full_stop_still_matches(self):
        answer, _ = apply_edits(DRAFT, [edit("Boost dose: 10.8 Gy in 6 fractions",
                                             "Do not deliver an additional 10.8 Gy; 55.8 Gy has been completed.")])
        # The old sentence's citation and full stop go with it, so [3] is not left on the new sentence
        self.assertEqual(answer.splitlines()[0], "- Do not deliver an additional 10.8 Gy; 55.8 Gy has been completed.")

    def test_a_split_replaces_the_whole_citation_group(self):
        answer, _ = apply_edits(DRAFT, [edit(
            "MRC CR07 used 25 Gy in 5 fractions and showed local control [5] [6].",
            "MRC CR07 used 25 Gy in 5 fractions [5]. Local control is not reported in the provided passages.",
            kind="split")])
        self.assertEqual(answer.splitlines()[1], "- MRC CR07 used 25 Gy in 5 fractions [5]. Local control is not "
                                                 "reported in the provided passages.")

    def test_a_citation_after_the_full_stop_goes_with_its_sentence(self):
        answer, _ = apply_edits("Use 30 Gy. [2] Then review.",
                                [edit("Use 30 Gy.", "Use 30 Gy in 10 fractions [4].", kind="recite")])
        self.assertEqual(answer, "Use 30 Gy in 10 fractions [4]. Then review.")

    def test_an_insert_goes_after_the_sentence_and_its_citations(self):
        answer, _ = apply_edits(DRAFT, [edit("Boost dose: 10.8 Gy in 6 fractions.",
                                             "Bladder D2cc should stay below 90 Gy EQD2 [7].",
                                             action="insert_after", kind="add")])
        self.assertEqual(answer.splitlines()[0], "- **Boost dose:** **10.8 Gy in 6 fractions** [3]. "
                                                 "Bladder D2cc should stay below 90 Gy EQD2 [7].")

    def test_a_quote_found_nowhere_changes_nothing_and_is_reported(self):
        answer, report = apply_edits(DRAFT, [edit("A sentence the answer does not have.", "Anything.")])
        self.assertEqual(answer, DRAFT)
        self.assertEqual(report["failed"][0]["why"], "not found")

    def test_a_quote_found_twice_changes_nothing(self):
        text = "Use 30 Gy [1]. Use 30 Gy [2]."
        answer, report = apply_edits(text, [edit("Use 30 Gy.", "Use 20 Gy.")])
        self.assertEqual(answer, text)
        self.assertEqual(report["failed"][0]["why"], "ambiguous")


PASSAGES = [{"citation_id": 1, "text": "Passage one says 45 Gy in 25 fractions."},
            {"citation_id": 2, "text": "Passage two."}]
REPLY = {"checklist": [{"item": "dose", "addressed": True, "gap": ""}],
         "edits": [edit("Boost 10.8 Gy.", "Do not boost."),
                   edit("X [1].", "Y [1]. Z.", kind="split"),
                   edit("Q.", "R.", action="sideways", kind="add")],
         "gaps": [GAP, {"item": "", "query": "no item"}, {"item": "target", "query": "q2"},
                  {"item": "technique", "query": "q3"}, {"item": "follow-up", "query": "q4"}],
         "main_recommendation_wrong": "false", "main_recommendation_reason": "", "confidence_score": 0.8}


class RecordingLLM:
    def __init__(self, reply):
        self.reply, self.prompt = reply, None

    def generate_structured(self, prompt, output_schema, system_prompt=None):
        self.prompt = prompt
        return self.reply


class EditCritiqueTests(unittest.TestCase):
    analysis = QueryAnalysis(original_query=QUESTION, intent=QueryIntent.TREATMENT_PROTOCOL)

    def critique(self):
        llm = RecordingLLM(REPLY)
        result = SelfCritic(llm, Config().agent, edits=True).critique(QUESTION, "Synthetic answer.", self.analysis,
                                                                     passages=PASSAGES)
        return result, " ".join(llm.prompt.split())

    def test_the_critique_reads_the_passages(self):
        _, prompt = self.critique()
        self.assertIn("[1] Passage one says 45 Gy in 25 fractions.", prompt)

    def test_a_named_gap_counts_only_if_no_passage_covers_it(self):
        _, prompt = self.critique()
        self.assertIn("or if it says the provided evidence does not cover the item and no passage covers it", prompt)

    def test_caveats_are_not_edits(self):
        _, prompt = self.critique()
        self.assertIn("Do not add caveats, reminders, counselling, or advice to seek review.", prompt)
        self.assertNotIn("Note any recommendation that could harm", prompt)

    def test_edits_are_kept_and_harm_makes_the_draft_unsafe(self):
        result, _ = self.critique()
        # The edit with an unknown action is dropped
        self.assertEqual([e["kind"] for e in result.edits], ["harm", "split"])
        self.assertFalse(result.is_safe)
        # ⚠️ "false" as text is not true
        self.assertFalse(result.main_recommendation_wrong)

    def test_gaps_come_with_a_search_query_three_at_most(self):
        result, prompt = self.critique()
        self.assertIn('List under "gaps"', prompt)
        # The gap with no item is dropped before the cap
        self.assertEqual([g["item"] for g in result.gaps], ["OAR constraints", "target", "technique"])

    def test_only_a_wrong_main_recommendation_asks_for_a_rewrite(self):
        critic = SelfCritic(None, Config().agent, edits=True)
        self.assertFalse(critic.should_retry(SelfCritiqueResult(is_complete=False, is_safe=False,
                                                                confidence_score=0.1)))
        self.assertTrue(critic.should_retry(SelfCritiqueResult(is_complete=True, is_safe=False, confidence_score=0.9,
                                                               main_recommendation_wrong=True,
                                                               concerns=["RT is not indicated."])))

    def test_the_default_critique_reads_no_passages(self):
        llm = RecordingLLM(REPLY)
        SelfCritic(llm, Config().agent).critique(QUESTION, "Synthetic answer.", self.analysis)
        self.assertNotIn("Passages", llm.prompt)


class GapFillerTests(unittest.TestCase):
    def test_a_sentence_is_written_only_for_this_patients_condition_and_setting(self):
        llm = RecordingLLM({"edits": [edit("The evidence does not give OAR limits.", "Sigmoid D2cc 70-75 Gy [10].",
                                           kind="fill"),
                                      edit("Q.", "R.", action="sideways", kind="fill")]})
        edits = GapFiller(llm).fill(QUESTION, "The evidence does not give OAR limits.", [GAP],
                                    "[10] Source: Gyn\nSigmoid D2cc 70-75 Gy.")
        prompt = " ".join(llm.prompt.split())
        self.assertIn("only if a new passage states it for this patient's condition and setting", prompt)
        self.assertIn("[10] Source: Gyn Sigmoid D2cc 70-75 Gy.", prompt)
        self.assertEqual([e["text"] for e in edits], ["Sigmoid D2cc 70-75 Gy [10]."])

    def test_an_unusable_reply_adds_nothing(self):
        self.assertEqual(GapFiller(RecordingLLM({"error": "bad JSON"})).fill(QUESTION, "A.", [GAP], "[10] x"), [])


def make_engine(**features):
    from core.agent import AgenticRAGEngine
    config = Config()
    for name, value in features.items():
        setattr(config.features, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        return AgenticRAGEngine(config=config, use_mock=True, enable_judge=False)


def saved_plan():
    analysis = QueryAnalysis(original_query=QUESTION, intent=QueryIntent.TREATMENT_PROTOCOL, is_complex=True)
    tasks = [SubTask(task_id=f"task_{i}", description=f"[Guidelines] topic {i}", query=f"saved query {i}",
                     retrieval_strategy=RetrievalStrategy.DENSE, top_k=6) for i in (1, 2)]
    return TaskPlan(plan_id="plan_saved", original_query=QUESTION, analysis=analysis, sub_tasks=tasks,
                    reasoning="Saved reasoning")


def first_sentence(answer):
    return answer.split(" for prompt:")[0]


def nothing(answer):
    return SelfCritiqueResult(is_complete=True, is_safe=True, confidence_score=0.8)


def edits_only(text="EDITED SENTENCE [1]."):
    return lambda answer: SelfCritiqueResult(is_complete=True, is_safe=True, confidence_score=0.8,
                                             edits=[edit(first_sentence(answer), text)])


def with_gaps(*gaps):
    return lambda answer: SelfCritiqueResult(is_complete=True, is_safe=True, confidence_score=0.8, gaps=list(gaps))


def wrong(answer):
    return SelfCritiqueResult(is_complete=True, is_safe=False, confidence_score=0.8, main_recommendation_wrong=True,
                              concerns=["RT is not indicated."], edits=[edit(first_sentence(answer), "EDITED [1].")])


def fills(answer_text="Sigmoid D2cc should stay at 70-75 Gy [2]."):
    return lambda query, answer, gaps, context: [edit(first_sentence(answer), answer_text, action="insert_after",
                                                      kind="fill")]


class EditFlowTests(unittest.TestCase):
    """The engine with critic_edits on. The mock writer's draft begins "Mock response #n" and cites nothing; the
    mock search finds one first-round passage, [1], and a search whose query starts "D2cc" finds one more."""

    def run_query(self, critiques, fill=None, **features):
        engine = make_engine(critic_edits=True, **features)
        self.searched, self.writes, self.given, self.filled = [], [], [], []
        search = engine.retriever.execute_task

        def searching(task, scope=None):
            self.searched.append(task.query)
            if not task.query.startswith("D2cc"):
                return search(task, scope)
            task.status = TaskStatus.COMPLETED
            task.retrieved_chunks = [{"faiss_id": 99, "chunk_id": 1, "text": "Sigmoid D2cc 70-75 Gy.",
                                      "path": "Gyn > Brachytherapy", "source_type": "review_notes"}]
            return task
        engine.retriever.execute_task = searching

        def critique(query, answer, analysis, passages=None):
            self.given.append([p["citation_id"] for p in passages or []])
            return critiques[len(self.given) - 1](answer)
        engine.self_critic.critique = critique
        filler = fill or (lambda query, answer, gaps, context: [])
        engine.gap_filler.fill = lambda *args: (self.filled.append(args), filler(*args))[1]
        write = engine.synthesizer.synthesize_multi_topic

        def recording(query, analysis, chunks_by_topic, **kwargs):
            self.writes.append(kwargs)
            return write(query, analysis, chunks_by_topic, **kwargs)
        engine.synthesizer.synthesize_multi_topic = recording
        with contextlib.redirect_stdout(io.StringIO()):
            return engine.query(QUESTION, plan=saved_plan())

    def test_edits_change_the_answer_without_a_rewrite_or_a_second_critique(self):
        response = self.run_query([edits_only()])
        self.assertEqual(len(self.writes), 1)
        self.assertIn("EDITED SENTENCE [1].", response.answer)
        self.assertEqual(self.given, [[1]])
        self.assertEqual(len(response.metadata["critic_edits"]["applied"]), 1)
        self.assertTrue(response.metadata["first_draft"].startswith("Mock response"))

    def test_a_passage_cited_by_an_edit_joins_the_source_list(self):
        response = self.run_query([edits_only()])
        self.assertEqual([c.citation_id for c in response.citations], [1])

    def test_a_wrong_main_recommendation_is_rewritten_once_and_critiqued_again(self):
        response = self.run_query([wrong, nothing], replan=True)
        self.assertEqual(len(self.writes), 2)
        self.assertIn("RT is not indicated.", self.writes[1]["feedback"])
        # No re-plan, even with the switch on
        self.assertEqual(self.searched, ["saved query 1", "saved query 2"])
        self.assertNotIn("EDITED", response.answer)
        self.assertTrue(response.metadata["critic_edits"]["rewritten"])
        self.assertEqual(len(self.given), 2)

    def test_a_verified_gap_is_searched_filled_and_critiqued_again(self):
        response = self.run_query([with_gaps(GAP), edits_only("SECOND PASS [2].")], fill=fills())
        # One search per source, as the planner pairs them
        self.assertEqual(self.searched.count("D2cc constraints"), 2)
        # The found passage is numbered after the first round's, and both passes and the filler see it
        self.assertEqual(self.given, [[1], [1, 2]])
        self.assertIn("[2] Source: Gyn > Brachytherapy", self.filled[0][3])
        report = response.metadata["critic_edits"]
        self.assertEqual(report["gap_passages"], [2])
        self.assertEqual(len(report["fill"]["applied"]), 1)
        self.assertEqual(len(report["second"]["applied"]), 1)
        self.assertIn("SECOND PASS [2].", response.answer)
        self.assertEqual([c.citation_id for c in response.citations], [2])
        self.assertEqual([p["citation_id"] for p in response.context_passages], [1, 2])

    def test_a_search_that_finds_nothing_new_writes_nothing(self):
        response = self.run_query([with_gaps({"item": "technique", "query": "known passages only"})])
        self.assertEqual(self.filled, [])
        self.assertEqual(len(self.given), 1)
        self.assertEqual(response.metadata["critic_edits"]["gap_passages"], [])

    def test_at_most_three_gaps_are_searched(self):
        gaps = [{"item": f"item {i}", "query": f"D2cc {i}"} for i in range(5)]
        self.run_query([with_gaps(*gaps)])
        self.assertEqual(sorted(set(q for q in self.searched if q.startswith("D2cc"))), ["D2cc 0", "D2cc 1", "D2cc 2"])

    def test_the_critique_can_run_on_its_own_model(self):
        engine = make_engine(critic_edits=True)
        self.assertIs(engine.self_critic.llm, engine.llm)
        engine = make_engine(critic_edits=True, critic_model="us.openai.gpt-5.6-sol")
        self.assertIsNot(engine.self_critic.llm, engine.llm)


if __name__ == "__main__":
    unittest.main()
