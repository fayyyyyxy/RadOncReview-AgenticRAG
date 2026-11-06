"""Which sentence each citation is checked against, without AWS."""

import re
import unittest

from core.citation_check import GROUP, CitationChecker, sentence_of


# The rewrite of GI-019 in gpt56terra-headings: [5] [6] follow a full stop, and another sentence follows on the line
MRC = ("MRC CR07 used 25 Gy in 5 fractions preoperatively and demonstrated that short-course preoperative RT "
       "effectively improves local control in operable rectal cancer.")
GUIDELINE = ("The guideline conditionally supports SCRT followed by chemotherapy in patients with increased "
             "local-recurrence risk.")
ANSWER = f"- {MRC} [5] [6] {GUIDELINE} [4]"
PASSAGES = [{"citation_id": n, "text": f"Passage {n} text"} for n in (4, 5, 6)]


class FakeLLM:
    """Records the prompt and rejects the items that cite one of the given passages."""

    def __init__(self, rejected=()):
        self.rejected = set(rejected)
        self.prompt = None

    def generate_structured(self, prompt, output_schema, system_prompt=None):
        self.prompt = prompt
        items = re.findall(r"^(\d+)\. Passage \[(\d+)\]", prompt, re.M)
        return {"verdicts": [{"item": i, "supported": "no" if int(n) in self.rejected else "yes"} for i, n in items]}


def item_lines(prompt, n):
    return [line for line in prompt.splitlines() if f"Passage [{n}] | Sentence:" in line]


class SentenceOfTests(unittest.TestCase):
    def test_citation_after_a_stop_is_checked_against_the_sentence_before_it(self):
        group = next(GROUP.finditer(ANSWER))
        sentence = sentence_of(ANSWER, group.start(), group.end())
        self.assertIn(MRC, sentence)
        self.assertNotIn(GUIDELINE, sentence)

    def test_citation_inside_a_sentence_keeps_the_whole_sentence(self):
        answer = "Treat to 50 Gy [3] in 25 fractions. Then reassess."
        group = next(GROUP.finditer(answer))
        self.assertEqual(sentence_of(answer, group.start(), group.end()), "Treat to 50 Gy [3] in 25 fractions.")

    def test_citation_at_the_end_of_a_line_is_unchanged(self):
        answer = "A synthetic fact. [2]\nNext line."
        group = next(GROUP.finditer(answer))
        self.assertEqual(sentence_of(answer, group.start(), group.end()), "A synthetic fact. [2]")


class CitationCheckerTests(unittest.TestCase):
    def test_each_citation_is_asked_about_with_its_own_sentence(self):
        llm = FakeLLM()
        checked, report = CitationChecker(llm).check(ANSWER, PASSAGES)

        for n in (5, 6):
            self.assertTrue(item_lines(llm.prompt, n))
            for line in item_lines(llm.prompt, n):
                self.assertIn(MRC, line)
                self.assertNotIn(GUIDELINE, line)
        self.assertTrue(all(GUIDELINE in line for line in item_lines(llm.prompt, 4)))
        self.assertEqual(checked, ANSWER)
        self.assertEqual(report["removed"], 0)

    def test_a_removed_citation_reports_only_its_own_sentence(self):
        checked, report = CitationChecker(FakeLLM(rejected={5, 6})).check(ANSWER, PASSAGES)

        self.assertEqual(checked, f"- {MRC} {GUIDELINE} [4]")
        self.assertEqual(report["removed"], 2)
        self.assertEqual(report["uncited_sentences"], [f"- {MRC}"])


if __name__ == "__main__":
    unittest.main()
