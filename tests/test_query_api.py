"""API progress, responsiveness, and request boundaries with a synthetic engine; no AWS."""
import asyncio
import json
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

import api
from core.models import AgentResponse, Citation
from core.site_routing import scope_for


class QueryAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gate_patch = patch.object(api, "_query_gate", asyncio.Lock())
        self.gate_patch.start()
        self.addCleanup(self.gate_patch.stop)
        self.entered, self.release = Event(), Event()
        self.addCleanup(self.release.set)
        self.calls = []
        self.engine = SimpleNamespace()

        def query(question, labels, **kwargs):
            self.calls.append((question, labels, kwargs))
            kwargs["progress_callback"]("evaluate")
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("Synthetic evaluator was not released")
            return AgentResponse(query=question, answer="Synthetic checked answer", latency_ms=1)

        self.engine.query = query
        self.agent_patch = patch.object(api, "get_agent", return_value=self.engine)
        self.agent_patch.start()
        self.addCleanup(self.agent_patch.stop)

    async def test_progress_arrives_and_health_responds_while_evaluation_is_blocked(self):
        request = api.QueryRequest(query="Follow-up?", disease_site="Breast", filter_by_site=True,
                                   prior_questions=["Initial synthetic question"])
        response = await api.query_with_progress(request)
        iterator = response.body_iterator
        events = []
        while True:
            event = json.loads(await asyncio.wait_for(anext(iterator), timeout=3))
            events.append(event)
            if event.get("stage") == "evaluate":
                break
        self.assertFalse(any(e["type"] == "result" for e in events))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://synthetic") as client:
            health = await asyncio.wait_for(client.get("/health"), timeout=1)
        self.assertEqual(health.status_code, 200)
        self.release.set()
        async for line in iterator:
            events.append(json.loads(line))
        result = events[-1]
        self.assertEqual(result["type"], "result")
        self.assertEqual(result["data"]["answer"], "Synthetic checked answer")
        kwargs = self.calls[0][2]
        self.assertFalse(kwargs["retain_history"])
        self.assertFalse(kwargs["run_evaluation"])
        self.assertEqual(kwargs["evidence_scope"], scope_for("Breast"))
        self.assertEqual(kwargs["prior_questions"], ["Initial synthetic question"])

    async def test_disconnected_stream_keeps_engine_gate_until_worker_finishes(self):
        response = await api.query_with_progress(api.QueryRequest(query="First question"))
        iterator = response.body_iterator
        while json.loads(await asyncio.wait_for(anext(iterator), timeout=3)).get("stage") != "evaluate":
            pass
        await iterator.aclose()
        second = asyncio.create_task(api._execute_query(api.QueryRequest(query="Second question"), api.CaseLabels()))
        await asyncio.sleep(0)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(api._query_gate.locked())
        self.release.set()
        await asyncio.wait_for(second, timeout=3)
        self.assertEqual([c[0] for c in self.calls], ["First question", "Second question"])

    async def test_json_route_still_returns_answer_and_rejects_invalid_context_and_category(self):
        self.release.set()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://synthetic") as client:
            response = await client.post("/query", json={"query": "Synthetic question"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answer"], "Synthetic checked answer")
            self.assertFalse(self.calls[-1][2]['run_evaluation'])
            for payload in [
                {"query": "q", "prior_questions": ["a", "b", "c"]},
                {"query": "q", "prior_questions": ["a" * 2001]},
                {"query": "q", "disease_site": "Unknown"},
            ]:
                invalid = await client.post("/query/stream", json=payload)
                self.assertEqual(invalid.status_code, 422)
            options = await client.get("/options")
            self.assertIn("Breast", options.json()["disease_sites"])
        self.assertEqual(len(self.calls), 1)

    async def test_quality_review_is_explicitly_enabled_per_request(self):
        self.release.set()
        request = api.QueryRequest(query='Synthetic quality-review question', run_evaluation=True)
        await api.query_guidelines(request)
        self.assertTrue(self.calls[0][2]['run_evaluation'])

    async def test_full_writer_passage_is_returned_without_using_truncated_citation(self):
        full_text = 'Synthetic full passage. ' * 100
        self.engine.query = lambda question, labels, **kwargs: AgentResponse(
            query=question, answer='Supported [1]',
            citations=[Citation(1, 'source', full_text[:300], 0.9, 7)],
            context_passages=[{'citation_id': 1, 'source_path': 'source', 'text': full_text}])
        result = await api.query_guidelines(api.QueryRequest(query='Synthetic question'))
        self.assertEqual(result.evidence_passages[0]['text'], full_text)
        self.assertEqual(len(result.citations[0].excerpt), 200)
        self.assertGreaterEqual(result.metadata['queue_ms'], 0)
        hidden = await api.query_guidelines(api.QueryRequest(query='q', include_citations=False))
        self.assertEqual(hidden.evidence_passages, [])

    async def test_editable_context_is_forwarded_and_oversized_context_is_rejected(self):
        self.release.set()
        await api.query_guidelines(api.QueryRequest(query='q', conversation_context='Corrected stage: II'))
        self.assertEqual(self.calls[0][2]['conversation_context'], 'Corrected stage: II')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://synthetic') as client:
            invalid = await client.post('/query', json={'query': 'q', 'conversation_context': 'x' * 4001})
            self.assertEqual(invalid.status_code, 422)

    async def test_stream_failure_returns_error_without_partial_answer(self):
        def fail(*args, **kwargs):
            raise RuntimeError("Synthetic failure")

        self.engine.query = fail
        response = await api.query_with_progress(api.QueryRequest(query="Synthetic question"))
        events = [json.loads(line) async for line in response.body_iterator]
        self.assertEqual(events[-1]["type"], "error")
        self.assertFalse(any(e["type"] == "result" for e in events))


if __name__ == "__main__":
    unittest.main()
