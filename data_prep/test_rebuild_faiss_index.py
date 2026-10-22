"""FAISS rebuild on synthetic rows with a fake embedding model; no AWS."""

import hashlib
import io
import json
import random
import time
import unittest

import faiss
import numpy as np

from rebuild_faiss_index import build, end_to_end_misses, stored_ids
from config import RetrieverConfig
from core.retriever import RetrieverManager

ROWS = {7: "alpha", 3: "beta", 12: "gamma", 0: "delta", 5: "alpha"}   # 5 duplicates 7, like the repeated document


def vector(text):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=16).astype("float32")


class FakeBedrock:
    def invoke_model(self, modelId, body):
        text = json.loads(body)["inputText"]
        time.sleep(random.random() / 100)   # threads finish out of order
        return {"body": io.BytesIO(json.dumps({"embedding": vector(text).tolist()}).encode())}


def fake_retriever():
    retriever = RetrieverManager.__new__(RetrieverManager)
    retriever.config = RetrieverConfig()
    retriever.config.similarity_threshold = -1
    retriever._ids_by_category = None
    retriever._get_embedding = lambda text: vector(text) / np.linalg.norm(vector(text))
    retriever._fetch_chunks_from_dynamodb = lambda ids: [{"FAISS_ID": i, "text": ROWS[int(i)]} for i in ids]
    return retriever


class RebuildTests(unittest.TestCase):
    def test_each_vector_is_stored_under_its_own_id(self):
        index = build(ROWS, FakeBedrock(), "model", workers=4)
        self.assertEqual(stored_ids(index), set(ROWS))
        for faiss_id, text in ROWS.items():
            expected = vector(text) / np.linalg.norm(vector(text))
            np.testing.assert_allclose(index.reconstruct(faiss_id), expected, rtol=1e-5)

    def test_end_to_end_check_catches_a_shifted_index(self):
        index = build(ROWS, FakeBedrock(), "model")
        self.assertEqual(end_to_end_misses(fake_retriever(), index, ROWS, sorted(ROWS)), [])
        # The January fault: vectors in table order but answering with the neighbouring row's ID
        ids = sorted(ROWS)
        shifted = faiss.IndexIDMap2(faiss.IndexFlatIP(16))
        vectors = np.array([vector(ROWS[i]) for i in ids])
        faiss.normalize_L2(vectors)
        shifted.add_with_ids(vectors, np.array(ids[1:] + ids[:1], dtype="int64"))
        self.assertTrue(end_to_end_misses(fake_retriever(), shifted, ROWS, ids))

    def test_plain_flat_index_answers_with_positions(self):
        self.assertEqual(stored_ids(faiss.IndexFlatIP(4)), set())


if __name__ == "__main__":
    unittest.main()
