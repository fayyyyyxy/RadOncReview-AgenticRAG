"""Citation numbering and context recording in the synthesizer, without AWS."""

import unittest

from config import Config
from core.models import QueryAnalysis, QueryIntent
from core.synthesizer import ResponseSynthesizer


class FakeLLM:
    """Records the prompt and returns a fixed answer."""

    def __init__(self, answer):
        self.answer = answer
        self.prompt = None

    def generate(self, prompt, system_prompt=None):
        self.prompt = prompt
        return self.answer


def chunk(text):
    return {"text": text, "source_type": "review_notes", "path": f"{text[:12]}.md", "similarity_score": 0.5}


class SynthesizerContextTests(unittest.TestCase):
    def setUp(self):
        self.analysis = QueryAnalysis(original_query="Synthetic question", intent=QueryIntent.GENERAL)

    def synthesizer(self, answer):
        llm = FakeLLM(answer)
        return ResponseSynthesizer(llm, Config()), llm

    def test_multi_topic_labels_match_citations(self):
        topics = {name: [chunk(f"{name} passage {i}") for i in range(1, 4)] for name in ("Topic A", "Topic B", "Topic C")}
        synth, llm = self.synthesizer("Synthetic fact [5].")
        _, citations, _, passages = synth.synthesize_multi_topic("Synthetic question", self.analysis, topics)

        # Each label appears once in what the LLM reads, so [5] has one meaning.
        for n in range(1, 10):
            self.assertEqual(llm.prompt.count(f"[{n}] Source:"), 1)
        self.assertEqual([p["citation_id"] for p in passages], list(range(1, 10)))
        self.assertEqual([c.citation_id for c in citations], [5])
        self.assertEqual(passages[4]["text"], "Topic B passage 2")
        self.assertEqual(passages[4]["topic"], "Topic B")
        self.assertEqual(citations[0].chunk_text, passages[4]["text"])

    def test_passages_keep_full_text(self):
        long_text = "x" * 800
        synth, _ = self.synthesizer("Synthetic fact [1].")
        _, citations, _, passages = synth.synthesize("Synthetic question", self.analysis, None, [chunk(long_text)])

        self.assertEqual(passages, [{"citation_id": 1, "source_path": chunk(long_text)["path"], "text": long_text}])
        self.assertEqual(citations[0].chunk_text, long_text[:500])

    def test_no_chunks_records_no_passages(self):
        synth, llm = self.synthesizer("unused")
        self.assertEqual(synth.synthesize("Synthetic question", self.analysis, None, [])[3], [])
        self.assertIsNone(llm.prompt)


if __name__ == "__main__":
    unittest.main()
