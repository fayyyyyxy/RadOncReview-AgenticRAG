"""The completeness template (docs/rag_improvement_plan_v2.md, E1): the writer answers under the seven parts of the
workbook's Completeness anchor and the critique checks those parts as well as what the question asks. No AWS."""

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
                "suggestions": []}


def chunk(text):
    return {"text": text, "source_type": "review_notes", "path": f"{text}.md", "similarity_score": 0.5}


def write(multi=True, **kwargs):
    llm = FakeLLM()
    synth = ResponseSynthesizer(llm, Config())
    if multi:
        synth.synthesize_multi_topic(QUESTION, ANALYSIS, {"Topic A": [chunk("Passage A")]}, **kwargs)
    else:
        synth.synthesize(QUESTION, ANALYSIS, None, [chunk("Passage A")], **kwargs)
    return llm


def critique(checklist=None):
    llm = FakeLLM(checklist)
    result = SelfCritic(llm, Config().agent).critique(QUESTION, "Synthetic answer [1].", ANALYSIS)
    return llm, result


class WriterTests(unittest.TestCase):
    def test_every_part_is_listed_in_order_in_both_prompts(self):
        for multi in (True, False):
            prompt = write(multi).prompt
            positions = [prompt.find(part) for part in PLAN_PARTS]
            self.assertTrue(all(p >= 0 for p in positions), prompt)
            self.assertEqual(positions, sorted(positions))

    def test_each_heading_is_capped_in_both_prompts(self):
        for multi in (True, False):
            self.assertIn("at most three sentences or bullet points", write(multi).prompt)

    def test_a_missing_part_is_supplied_marked_as_not_in_the_evidence(self):
        # The judge scores a part named as missing as missing (E1 pilot, BEN-005), and a correct statement that is
        # not in the passages is not a hallucination to it
        for multi in (True, False):
            prompt = write(multi).prompt
            self.assertIn('give the usual practice without a citation and mark it "(not in the provided evidence)"',
                          prompt)
            self.assertNotIn("do not supply it from general knowledge", prompt)
            self.assertIn('"Not applicable"', prompt)

    def test_a_missing_case_detail_is_answered_under_a_stated_assumption(self):
        # The GU-034 pilot rewrite gave no boost dose because the case does not give the prior fractionation
        for multi in (True, False):
            self.assertIn("state the assumption, answer under it, and say what to confirm", write(multi).prompt)

    def test_no_instruction_contradicts_the_template(self):
        for multi in (True, False):
            llm = write(multi)
            self.assertNotIn("avoid tangential information", llm.system_prompt)
            self.assertNotIn("only include what's necessary to answer the question", llm.system_prompt)
            self.assertNotIn("aim for 3-5 sentences", llm.prompt)

    def test_a_rewrite_keeps_the_template_and_the_feedback(self):
        prompt = write(feedback="Address these missing aspects: Follow-up", draft="Earlier draft.").prompt
        self.assertIn(PLAN_PARTS[0], prompt)
        self.assertIn("Reviewer feedback: Address these missing aspects: Follow-up", prompt)


class CritiqueTests(unittest.TestCase):
    def test_every_part_is_checked_with_a_way_out_for_parts_that_do_not_apply(self):
        llm, _ = critique()
        for part in PLAN_PARTS:
            self.assertIn(part, llm.prompt)
        self.assertNotIn("Do not add anything the question does not ask about.", llm.prompt)
        self.assertIn("not applicable", llm.prompt)

    def test_a_part_left_unaddressed_sends_the_answer_back(self):
        checklist = [{"item": "dose", "addressed": True, "gap": ""},
                     {"item": "Follow-up", "addressed": False, "gap": "no surveillance plan"}]
        _, result = critique(checklist)
        self.assertEqual(result.missing_aspects, ["Follow-up: no surveillance plan"])
        self.assertTrue(SelfCritic(FakeLLM(), Config().agent).should_retry(result))


if __name__ == "__main__":
    unittest.main()
