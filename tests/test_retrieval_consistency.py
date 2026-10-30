"""Order-invariant ranking and full-passage deduplication; no AWS."""
import unittest
from unittest.mock import Mock

import numpy as np

from core.agent import AgenticRAGEngine
from core.models import RetrievedChunk
from core.retrieval_identity import passage_identity
from core.retriever import CrossEncoderReranker


class RankingTests(unittest.TestCase):
    def chunk(self, fid):
        return RetrievedChunk(faiss_id=fid, chunk_id=fid, text=f"Text {fid}", path=f"source {fid}",
                              category="Synthetic", similarity_score=0.5)

    def test_equal_scores_have_same_order_despite_backend_order(self):
        reranker = CrossEncoderReranker()
        reranker._model = Mock()
        reranker._model.predict.side_effect = lambda pairs, **kwargs: np.zeros(len(pairs))
        rows = [self.chunk(1), self.chunk(2), self.chunk(3)]
        first = [c.faiss_id for c in reranker.rerank("q", rows, 2)]
        second = [c.faiss_id for c in reranker.rerank("q", list(reversed(rows)), 2)]
        self.assertEqual(first, second)
        self.assertEqual(reranker._model.predict.call_args_list[0].args[0],
                         reranker._model.predict.call_args_list[1].args[0])

    def test_whole_passage_identity_does_not_merge_shared_opening_text(self):
        first = {"source_type": "clinical_trial", "s3_location": "s3://bucket/paper", "text": "x" * 100 + "first ending"}
        second = {**first, "text": "x" * 100 + "different ending"}
        self.assertNotEqual(passage_identity(first), passage_identity(second))
        self.assertNotEqual(AgenticRAGEngine._chunk_key(first), AgenticRAGEngine._chunk_key(second))


if __name__ == "__main__":
    unittest.main()
