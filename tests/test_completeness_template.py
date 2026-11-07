"""The completeness template (docs/rag_improvement_plan_v2.md, E1): with completeness_template on, the writer answers
under the seven parts of the workbook's Completeness anchor and the critique checks those parts as well as what the
question asks. Off, every prompt is the one the tag sends. No AWS."""

import contextlib
import io
import unittest

from config import Config
from core.models import QueryAnalysis, QueryIntent
from core.synthesizer import PLAN_PARTS, ResponseSynthesizer, SelfCritic

QUESTION = "Synthetic question"
ANALYSIS = QueryAnalysis(original_query=QUESTION, intent=QueryIntent.TREATMENT_PROTOCOL)


class FakeLLM:
    """Records what it was sent; returns a fixed answer, or a fixed checklist when asked for JSON."""

    def __init__(self, checklist=None):
        self.prompt = self.system_prompt = None
        self.checklist = checklist or [{"item": "dose", "addressed": True, "gap": ""}]

    def generate(self, prompt, system_prompt=None):
        self.prompt, self.system_prompt = prompt, system_prompt
        return "Synthetic answer [1]."

    def generate_structured(self, prompt, output_schema, system_prompt=None):
        self.prompt, self.system_prompt = prompt, system_prompt
        return {"checklist": self.checklist, "is_safe": True, "concerns": [], "confidence_score": 0.8,
                "suggestions": [], "edits": [], "gaps": [], "main_recommendation_wrong": False}


def chunk(text):
    return {"text": text, "source_type": "review_notes", "path": f"{text}.md", "similarity_score": 0.5}


def write(template, multi=True, **kwargs):
    config = Config()
    config.features.completeness_template = template
    llm = FakeLLM()
    synth = ResponseSynthesizer(llm, config)
    if multi:
        synth.synthesize_multi_topic(QUESTION, ANALYSIS, {"Topic A": [chunk("Passage A")]}, **kwargs)
    else:
        synth.synthesize(QUESTION, ANALYSIS, None, [chunk("Passage A")], **kwargs)
    return llm


def critique(template, edits=False, checklist=None):
    llm = FakeLLM(checklist)
    result = SelfCritic(llm, Config().agent, edits=edits, template=template).critique(
        QUESTION, "Synthetic answer [1].", ANALYSIS, passages=[{"citation_id": 1, "text": "Passage A"}])
    return llm, result


class WriterTests(unittest.TestCase):
    def test_off_sends_the_tags_prompts(self):
        for multi in (True, False):
            llm = write(False, multi)
            self.assertEqual(llm.system_prompt, ResponseSynthesizer.SYSTEM_PROMPT)
            self.assertNotIn("(not in the provided evidence)", llm.prompt)
            self.assertNotIn("state the assumption", llm.prompt)
            self.assertNotIn(PLAN_PARTS[-1], llm.prompt)

    def test_on_lists_every_part_in_order_in_both_prompts(self):
        for multi in (True, False):
            prompt = write(True, multi).prompt
            positions = [prompt.find(part) for part in PLAN_PARTS]
            self.assertTrue(all(p >= 0 for p in positions), prompt)
            self.assertEqual(positions, sorted(positions))

    def test_on_caps_each_heading_in_both_prompts(self):
        for multi in (True, False):
            self.assertIn("at most three sentences or bullet points", write(True, multi).prompt)

    def test_on_supplies_a_missing_part_marked_as_not_in_the_evidence(self):
        # The judge scores a part named as missing as missing (E1 pilot, BEN-005), and a correct statement that is
        # not in the passages is not a hallucination to it
        for multi in (True, False):
            prompt = write(True, multi).prompt
            self.assertIn('give the usual practice without a citation and mark it "(not in the provided evidence)"',
                          prompt)
            self.assertNotIn("do not supply it from general knowledge", prompt)
            self.assertIn('"Not applicable"', prompt)

    def test_on_answers_under_a_stated_assumption_instead_of_declining(self):
        # The GU-034 pilot rewrite gave no boost dose because the case does not give the prior fractionation
        for multi in (True, False):
            self.assertIn("state the assumption, answer under it, and say what to confirm", write(True, multi).prompt)

    def test_on_drops_the_instructions_that_contradict_the_template(self):
        for multi in (True, False):
            llm = write(True, multi)
            self.assertNotIn("avoid tangential information", llm.system_prompt)
            self.assertNotIn("only include what's necessary to answer the question", llm.system_prompt)
            self.assertNotIn("aim for 3-5 sentences", llm.prompt)

    def test_a_rewrite_keeps_the_template_and_the_feedback(self):
        prompt = write(True, feedback="Address these missing aspects: Follow-up", draft="Earlier draft.").prompt
        self.assertIn(PLAN_PARTS[0], prompt)
        self.assertIn("Reviewer feedback: Address these missing aspects: Follow-up", prompt)


class CritiqueTests(unittest.TestCase):
    def test_off_checks_only_what_the_question_asks(self):
        llm, _ = critique(False)
        self.assertNotIn(PLAN_PARTS[-1], llm.prompt)
        self.assertIn("Do not add anything the question does not ask about.", llm.prompt)

    def test_on_adds_every_part_and_a_way_out_for_parts_that_do_not_apply(self):
        llm, _ = critique(True)
        for part in PLAN_PARTS:
            self.assertIn(part, llm.prompt)
        self.assertNotIn("Do not add anything the question does not ask about.", llm.prompt)
        self.assertIn("not applicable", llm.prompt)
        # The tag's rule stays: saying the sources do not cover a part does not address it
        self.assertIn(SelfCritic.ADDRESSED_RULE, llm.prompt)

    def test_on_a_part_left_unaddressed_sends_the_answer_back(self):
        checklist = [{"item": "dose", "addressed": True, "gap": ""},
                     {"item": "Follow-up", "addressed": False, "gap": "no surveillance plan"}]
        _, result = critique(True, checklist=checklist)
        self.assertEqual(result.missing_aspects, ["Follow-up: no surveillance plan"])
        self.assertTrue(SelfCritic(FakeLLM(), Config().agent, template=True).should_retry(result))

    def test_the_critic_edits_prompt_is_not_changed(self):
        off, _ = critique(False, edits=True)
        on, _ = critique(True, edits=True)
        self.assertEqual(off.prompt, on.prompt)


class WiringTests(unittest.TestCase):
    def test_the_template_is_on_by_default(self):
        # The user's decision after the 40-case arm, although its Harm >= 3 bar failed (docs/rag_improvement_plan_v2.md)
        self.assertTrue(Config().features.completeness_template)

    def test_the_switch_reaches_the_writer_and_the_critique(self):
        from core.agent import AgenticRAGEngine
        for value in (False, True):
            config = Config()
            config.features.completeness_template = value
            with contextlib.redirect_stdout(io.StringIO()):
                engine = AgenticRAGEngine(config=config, use_mock=True, enable_judge=False)
            self.assertEqual(engine.self_critic.template, value)
            self.assertEqual(engine.synthesizer.config.features.completeness_template, value)


if __name__ == "__main__":
    unittest.main()
