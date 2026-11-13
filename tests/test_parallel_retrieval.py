"""Bounded retrieval concurrency, real batch reads, and explicit context; no AWS calls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from decimal import Decimal
import io
from threading import Barrier, Event, RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from config import AWSConfig, Config
from core.agent import AgenticRAGEngine
from core.models import SubTask, RetrievalStrategy, TaskStatus
from core.retriever import RetrieverManager
from core.site_routing import scope_for

class BatchReadTests(unittest.TestCase):
    def setUp(self):
        self.retriever = RetrieverManager.__new__(RetrieverManager)
        self.retriever.aws_config = AWSConfig(dynamodb_table="synthetic")
        self.retriever.dynamodb = Mock()
        self.retriever._metadata_lock = RLock()

    def test_batches_deduplicated_native_ids_and_restores_order_after_partial_retry(self):
        self.retriever.dynamodb.batch_get_item.side_effect = [
            {"Responses": {"synthetic": [{"FAISS_ID": Decimal(3), "score": Decimal("0.5")}]},
             "UnprocessedKeys": {"synthetic": {"Keys": [{"FAISS_ID": 1}]}}},
            {"Responses": {"synthetic": [{"FAISS_ID": Decimal(1)}]}},
        ]
        with patch("core.retriever.time.sleep"):
            rows = self.retriever._fetch_chunks_from_dynamodb([np.int64(1), 3, 1])
        self.assertEqual([r["FAISS_ID"] for r in rows], [1, 3])
        self.assertEqual(rows[1]["score"], 0.5)
        calls = self.retriever.dynamodb.batch_get_item.call_args_list
        self.assertEqual(calls[0].kwargs["RequestItems"]["synthetic"]["Keys"], [{"FAISS_ID": 1}, {"FAISS_ID": 3}])
        self.assertEqual(calls[1].kwargs["RequestItems"]["synthetic"]["Keys"], [{"FAISS_ID": 1}])

    def test_splits_at_service_limit_and_handles_missing_items(self):
        self.retriever.dynamodb.batch_get_item.return_value = {"Responses": {"synthetic": []}}
        self.assertEqual(self.retriever._fetch_chunks_from_dynamodb(range(101)), [])
        calls = self.retriever.dynamodb.batch_get_item.call_args_list
        self.assertEqual([len(c.kwargs["RequestItems"]["synthetic"]["Keys"]) for c in calls], [100, 1])

    def test_incomplete_reads_fail_after_bounded_retries(self):
        self.retriever.dynamodb.batch_get_item.return_value = {
            "UnprocessedKeys": {"synthetic": {"Keys": [{"FAISS_ID": 1}]}}}
        with patch("core.retriever.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.retriever._fetch_chunks_from_dynamodb([1])
        self.assertEqual(self.retriever.dynamodb.batch_get_item.call_count, 5)
        self.assertEqual(sleep.call_count, 4)

    def test_empty_selection_does_not_call_database(self):
        self.assertEqual(self.retriever._fetch_chunks_from_dynamodb([]), [])
        self.retriever.dynamodb.batch_get_item.assert_not_called()


class ParallelSearchTests(unittest.TestCase):
    def test_overlapping_searches_keep_plan_order_and_scope(self):
        engine = AgenticRAGEngine(Config(), use_mock=True, enable_judge=False)
        engine.config.retriever.max_workers = 2
        tasks = [SubTask(str(i), str(i), "q", RetrievalStrategy.DENSE) for i in range(2)]
        barrier = Barrier(2)
        second_done = Event()
        scope = scope_for("Breast")

        def search(task, received_scope):
            self.assertIs(received_scope, scope)
            barrier.wait(timeout=3)  # would fail if the searches did not overlap
            if task.task_id == "0":
                self.assertTrue(second_done.wait(timeout=3))
            else:
                second_done.set()
            task.status = TaskStatus.COMPLETED
            task.result = task.task_id
            task.retrieved_chunks = [{"text": "passage " + task.task_id}]

        engine.retriever.execute_task = search
        with redirect_stdout(io.StringIO()):
            engine._execute_retrieval_tasks(tasks, scope)
        self.assertEqual([t.task_id for t in tasks], ["0", "1"])
        self.assertEqual([t.retrieved_chunks[0]["text"] for t in tasks], ["passage 0", "passage 1"])
        self.assertIn("Task '0'", engine.reasoning_trace[0])
        self.assertIn("Task '1'", engine.reasoning_trace[1])

    def test_concurrent_cold_searches_download_and_publish_one_index(self):
        retriever = RetrieverManager.__new__(RetrieverManager)
        retriever.aws_config = AWSConfig(s3_bucket="synthetic")
        retriever.s3 = Mock()
        retriever._index_lock = RLock()
        retriever._faiss_index_loaded = False
        index = SimpleNamespace(ntotal=2)
        with patch("core.retriever.faiss.read_index", return_value=index), redirect_stdout(io.StringIO()):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: retriever._load_faiss_index(), range(2)))
        self.assertEqual(results, [index, index])
        retriever.s3.download_file.assert_called_once()
