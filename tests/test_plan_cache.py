"""Expiry, real on-disk budget, LRU/old-version eviction, concurrency and audit retention."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from config import Config
from core.plan_cache import PlanCache, encode
from core.models import QueryAnalysis, QueryIntent
from core.versioned_planner import VersionedPlanner

IDENTITY = {'version': 'test_v1', 'code_sha256': 'current', 'prompt_sha256': 'prompt'}


class PlanCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = [1_800_000_000.0]
        self.cache = PlanCache(self.root/'cache', audit_folder=self.root/'audits', clock=lambda: self.clock[0])

    def record(self, name, identity=None, padding=800, created=None):
        identity = dict(IDENTITY if identity is None else identity)
        key = hashlib.sha256(name.encode()).hexdigest()
        return key, {'cache_key': key, 'identity': identity, 'payload': {'text': 'x'*padding},
                     'created_at': datetime.fromtimestamp(self.clock[0] if created is None else created, timezone.utc).isoformat()}

    def store(self, name, **kwargs):
        key, record = self.record(name, **kwargs)
        result = self.cache.store(key, record, IDENTITY)
        self.assertTrue(result['stored'])
        return key

    def disk_bytes(self):
        return sum(p.stat().st_size for p in self.cache.folder.iterdir() if p.is_file())

    def budget_for(self, keys):
        # Existing index overestimates the future, smaller index after removing entries.
        return sum((self.cache.folder/(k+'.json')).stat().st_size for k in keys) + self.cache.index_path.stat().st_size

    def test_default_limit_is_100_megabytes_and_thirty_days(self):
        stats = self.cache.inspect(IDENTITY)
        self.assertEqual(stats['max_bytes'], 100_000_000)
        self.assertEqual(stats['ttl_days'], 30)

    def test_expiry_is_from_creation_and_hot_reads_cannot_extend_it(self):
        key = self.store('expires')
        self.clock[0] += 30*86400-1
        self.assertIsNotNone(self.cache.load(key, IDENTITY))
        self.clock[0] += 1
        self.assertIsNone(self.cache.load(key, IDENTITY))
        self.assertFalse((self.cache.folder/(key+'.json')).exists())

    def test_real_disk_budget_includes_index_and_uses_lru(self):
        a = self.store('a'); self.clock[0] += 10
        b = self.store('b'); self.clock[0] += 10
        c = self.store('c'); self.clock[0] += 10
        self.cache.load(a, IDENTITY)
        self.cache.max_bytes = self.budget_for([a, c])
        result = self.cache.maintain(IDENTITY)
        self.assertEqual(result['evicted'], 1)
        self.assertIsNone(self.cache.load(b, IDENTITY))
        self.assertIsNotNone(self.cache.load(a, IDENTITY))
        self.assertIsNotNone(self.cache.load(c, IDENTITY))
        self.assertLessEqual(self.disk_bytes(), self.cache.max_bytes)

    def test_old_revision_is_evicted_before_older_current_plan(self):
        a = self.store('current-cold'); self.clock[0] += 10
        old = self.store('old-hot', identity={**IDENTITY, 'code_sha256': 'obsolete'})
        self.clock[0] += 10
        b = self.store('current-hot')
        self.cache.load(old, IDENTITY)
        self.cache.max_bytes = self.budget_for([a, b])
        result = self.cache.maintain(IDENTITY)
        self.assertEqual(result['obsolete_evicted'], 1)
        self.assertIsNone(self.cache.load(old, IDENTITY))
        self.assertIsNotNone(self.cache.load(a, IDENTITY))

    def test_different_model_configuration_is_not_itself_an_old_code_revision(self):
        a = self.store('older-current-model', identity={**IDENTITY, 'model': {'model_id': 'A'}})
        self.clock[0] += 10
        b = self.store('hot-other-model', identity={**IDENTITY, 'model': {'model_id': 'B'}})
        self.cache.max_bytes = self.budget_for([b])
        result = self.cache.maintain(IDENTITY)
        self.assertEqual(result['obsolete_evicted'], 0)
        self.assertIsNone(self.cache.load(a, IDENTITY))
        self.assertIsNotNone(self.cache.load(b, IDENTITY))

    def test_oversized_record_is_not_cached_and_does_not_displace_existing_plan(self):
        key = self.store('keep')
        before = (self.cache.folder/(key+'.json')).read_bytes()
        self.cache.max_bytes = self.disk_bytes()+500
        large, record = self.record('oversize', padding=10_000)
        result = self.cache.store(large, record, IDENTITY)
        self.assertFalse(result['stored'])
        self.assertEqual(result['reason'], 'record_exceeds_budget')
        self.assertEqual((self.cache.folder/(key+'.json')).read_bytes(), before)
        self.assertLessEqual(self.disk_bytes(), self.cache.max_bytes)

    def test_oversized_alias_update_preserves_source_and_budget(self):
        key = self.store('source')
        self.cache.max_bytes = self.disk_bytes()+200
        def update(record):
            record['paraphrases'] = ['x'*10_000]
            return record
        result = self.cache.update(key, IDENTITY, update)
        self.assertFalse(result['stored'])
        self.assertTrue(result['available'])
        self.assertNotIn('paraphrases', self.cache.load(key, IDENTITY))
        self.assertLessEqual(self.disk_bytes(), self.cache.max_bytes)

    def test_failed_alias_admission_still_bounds_growing_access_metadata(self):
        self.clock[0] = 1.0
        key = self.store('tight-metadata-budget')
        self.cache.max_bytes = self.disk_bytes()
        self.clock[0] = 10001.0
        self.cache.update(key, IDENTITY, lambda record: {**record, 'paraphrases': ['x'*10_000]})
        self.assertLessEqual(self.disk_bytes(), self.cache.max_bytes)

    def test_explicit_audit_survives_expiration_and_eviction_byte_for_byte(self):
        key = self.store('retain')
        audit = self.cache.retain(key, note='Intentionally kept for review')
        before = audit.read_bytes()
        self.clock[0] += 31*86400
        self.cache.maintain(IDENTITY)
        self.assertIsNone(self.cache.load(key, IDENTITY))
        self.cache.max_bytes = 1
        self.cache.maintain(IDENTITY)
        self.assertEqual(audit.read_bytes(), before)
        self.assertEqual(json.loads(before)['source_record']['cache_key'], key)
        self.assertEqual(audit.stat().st_mode & 0o777, 0o600)
        self.assertFalse(self.cache.folder in audit.parents)

    def test_routine_cache_use_does_not_create_an_unbounded_audit_archive(self):
        self.store('disposable')
        self.cache.maintain(IDENTITY)
        self.assertFalse(self.cache.audit_folder.exists())

    def test_audit_and_cache_paths_cannot_overlap(self):
        for path in [self.root/'cache', self.root/'cache'/'audits', self.root]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                PlanCache(self.root/'cache', audit_folder=path)

    def test_legacy_record_uses_creation_date_not_recent_file_copy_time(self):
        self.cache.folder.mkdir()
        key, record = self.record('legacy', created=self.clock[0]-45*86400)
        path = self.cache.folder/(key+'.json')
        path.write_bytes(encode(record))
        preview = self.cache.inspect(IDENTITY)
        self.assertEqual(preview['expired_plans'], 1)
        self.assertTrue(path.exists())
        audit = self.cache.retain(key)
        self.cache.maintain(IDENTITY)
        self.assertFalse(path.exists())
        self.assertTrue(audit.exists())

    def test_index_can_be_rebuilt_without_losing_current_records(self):
        key = self.store('survives')
        self.cache.index_path.write_text('bad index')
        self.cache.maintain(IDENTITY)
        self.assertIsNotNone(self.cache.load(key, IDENTITY))
        self.assertEqual(json.loads(self.cache.index_path.read_text())['schema'], 1)

    def test_unrelated_files_are_not_deleted(self):
        self.cache.folder.mkdir()
        extra = self.cache.folder/'human-review.json'
        extra.write_text('keep me')
        self.cache.max_bytes = 1
        self.cache.maintain(IDENTITY)
        self.assertEqual(extra.read_text(), 'keep me')
        self.assertEqual(self.cache.inspect(IDENTITY)['unmanaged_bytes'], 7)

    def test_abandoned_cache_writes_are_cleaned_but_inspection_does_not_delete_them(self):
        key = self.store('live')
        abandoned = self.cache.folder/('.'+key+'.json.'+'a'*32+'.tmp')
        abandoned.write_text('unfinished cache write')
        self.cache.inspect(IDENTITY)
        self.assertTrue(abandoned.exists())
        self.cache.maintain(IDENTITY)
        self.assertFalse(abandoned.exists())
        self.assertIsNotNone(self.cache.load(key, IDENTITY))

    def test_concurrent_writers_and_cleanup_remain_within_the_disk_budget(self):
        self.cache.max_bytes = 5000
        def work(i):
            key, record = self.record('thread-'+str(i))
            self.cache.store(key, record, IDENTITY)
            self.cache.maintain(IDENTITY)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(work, i) for i in range(20)]
            for future in futures:
                future.result(timeout=10)
        self.assertLessEqual(self.disk_bytes(), self.cache.max_bytes)
        self.assertEqual(self.cache.inspect(IDENTITY)['bytes_used'], self.disk_bytes())

    def test_invalid_limits_are_rejected(self):
        for limit in [0, -1, True, 2.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                PlanCache(self.root/'cache', max_bytes=limit)
        for ttl in [0, -1, True, None, float('inf'), float('nan')]:
            with self.subTest(ttl=ttl), self.assertRaises(ValueError):
                PlanCache(self.root/'cache', ttl_days=ttl)


class PlannerStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = Config()
        self.config.features.plan_cache_path = str(Path(self.tmp.name)/'cache')
        self.config.features.plan_audit_path = str(Path(self.tmp.name)/'audits')
        self.llm = Mock()
        self.llm.generate_structured.return_value = {
            'requirements': [{'id': 'R1', 'kind': 'explicit', 'need': 'Synthetic information need',
                              'anchors': [{'source': 'question', 'quote': 'information'}]}],
            'topics': [{'requirement_ids': ['R1'], 'search_terms': ['synthetic information'],
                        'context_ids': ['C1'], 'context_reason': 'Original context.'}],
            'reasoning': 'Investigate the requested information.'}
        self.analysis = QueryAnalysis('What information is needed?', QueryIntent.GENERAL)

    def test_oversized_valid_plan_is_still_returned_and_can_be_explicitly_retained(self):
        self.config.features.plan_cache_max_bytes = 100
        planner = VersionedPlanner(self.llm, self.config)
        plan = planner.plan(self.analysis)
        self.assertEqual(plan.planning['cache_status'], 'created_not_cached')
        self.assertEqual(len(plan.sub_tasks), 2)
        plan.sub_tasks[0].retrieved_chunks = [{'text': 'Do not put evidence into a planning audit'}]
        audit = planner.retain_for_review(plan, 'Review this uncached plan')
        saved = json.loads(audit.read_text())
        self.assertIsNone(saved['source_record'])
        self.assertEqual(saved['request_plan']['original_query'], self.analysis.original_query)
        self.assertNotIn('Do not put evidence', audit.read_text())

    def test_expired_plan_is_generated_again_instead_of_reused(self):
        planner = VersionedPlanner(self.llm, self.config)
        first = planner.plan(self.analysis)
        path = planner.folder/(first.planning['cache_key']+'.json')
        record = json.loads(path.read_text())
        record['created_at'] = '2000-01-01T00:00:00+00:00'
        path.write_text(json.dumps(record))
        second = planner.plan(self.analysis)
        self.assertEqual(second.planning['cache_status'], 'created')
        self.assertEqual(self.llm.generate_structured.call_count, 2)


if __name__ == '__main__':
    unittest.main()
