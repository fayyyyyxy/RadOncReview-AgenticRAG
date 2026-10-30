"""Paraphrases reuse search concepts while keeping current source facts and anchors."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock

from config import Config
from core.models import QueryAnalysis, QueryIntent
from core.versioned_planner import VersionedPlanner, context_units, input_sources
from core.plan_paraphrases import protected_details

OLD = ('Patient vignette:\nA 60-year-old woman has RIGHT breast cancer. No previous RT.'
       '\n\nClinical question:\nWhat dose and technique are appropriate?')
NEW = ('Patient vignette:\nA woman aged 60 has cancer of the RIGHT breast. She has never received RT.'
       '\n\nClinical question:\nWhich radiation dose and treatment technique should be used?')


def plan_payload():
    return {'requirements': [{'id': 'R1', 'kind': 'explicit', 'need': 'Radiotherapy dose',
                              'anchors': [{'source': 'question', 'quote': 'dose'}]},
                             {'id': 'R2', 'kind': 'explicit', 'need': 'Radiotherapy technique',
                              'anchors': [{'source': 'question', 'quote': 'technique'}]}],
            'topics': [{'requirement_ids': [rid], 'search_terms': [term], 'context_ids': ['C1', 'C2'],
                        'context_reason': 'Patient condition and prior treatment.'}
                       for rid, term in [('R1', 'radiotherapy dose'), ('R2', 'radiotherapy technique')]],
            'reasoning': 'Investigate both requested needs.'}


class ParaphraseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config()
        self.cfg.features.plan_cache_path = self.tmp.name
        self.llm = Mock()
        self.llm.generate_structured.return_value = plan_payload()
        self.planner = VersionedPlanner(self.llm, self.cfg)
        self.original = self.planner.plan(self.analysis(OLD))
        self.llm.reset_mock()

    def analysis(self, query):
        return QueryAnalysis(query, QueryIntent.GENERAL)

    def proof(self, query=NEW):
        units = context_units(input_sources(query))
        return {'equivalent': True, 'candidate_key': self.original.planning['cache_key'],
                'facts_unchanged': True, 'needs_unchanged': True, 'uncertainty_unchanged': True,
                'reason': 'Same explicit patient facts and dose/technique requests.',
                'anchors': [{'requirement_id': 'R1', 'anchors': [{'source': 'question', 'quote': 'dose'}]},
                            {'requirement_id': 'R2', 'anchors': [{'source': 'question', 'quote': 'technique'}]}],
                'unit_mappings': [{'old_id': f'C{i}', 'new_ids': [u['id']], 'equivalent': True}
                                  for i, u in enumerate(units, 1)]}

    def test_reanchors_paraphrase_without_regenerating_topics_or_copying_old_context(self):
        self.llm.generate_structured.return_value = self.proof()
        result = self.planner.plan(self.analysis(NEW))
        self.assertEqual(result.planning['cache_status'], 'paraphrase_hit')
        self.assertEqual(result.planning['llm_calls'], 1)
        self.assertEqual(result.planning['cache_source_key'], self.original.planning['cache_key'])
        self.assertEqual(result.original_query, NEW)
        self.assertNotEqual(result.planning['cache_key'], self.original.planning['cache_key'])
        self.assertEqual([t['search_terms'] for t in result.planning['payload']['topics']],
                         [t['search_terms'] for t in self.original.planning['payload']['topics']])
        self.assertTrue(all('A woman aged 60' in t.query and 'She has never received RT.' in t.query for t in result.sub_tasks))
        self.assertTrue(all('60-year-old' not in t.query for t in result.sub_tasks))
        self.assertEqual(len(list(Path(self.tmp.name).glob('*.json'))), 1)

    def test_verified_wording_repeats_without_any_llm_call(self):
        self.llm.generate_structured.return_value = self.proof()
        first = self.planner.plan(self.analysis(NEW))
        self.llm.reset_mock()
        self.llm.generate_structured.side_effect = AssertionError('Saved alias must not use an LLM')
        repeated = self.planner.plan(self.analysis(NEW))
        self.llm.generate_structured.assert_not_called()
        self.assertEqual(repeated.planning['cache_status'], 'paraphrase_alias_hit')
        self.assertEqual(repeated.planning['payload'], first.planning['payload'])
        self.assertNotEqual(repeated.sub_tasks[0].task_id, first.sub_tasks[0].task_id)
        self.assertTrue(all(not t.retrieved_chunks for t in repeated.sub_tasks))

    def test_changed_protected_facts_skip_equivalence_check(self):
        variants = [NEW.replace('60', '61'), NEW.replace('RIGHT', 'LEFT'),
                    NEW.replace('woman', 'man'), NEW.replace('never received RT', 'received RT'),
                    NEW.replace('never received RT', 'never received RT; dose is unknown')]
        for query in variants:
            with self.subTest(query=query):
                self.llm.reset_mock()
                self.llm.generate_structured.return_value = plan_payload()
                result = self.planner.plan(self.analysis(query))
                self.assertEqual(result.planning['cache_status'], 'created')
                self.assertEqual(result.planning['paraphrase_check_calls'], 0)
                self.llm.generate_structured.assert_called_once()

    def test_protected_details_preserve_numeric_roles_and_tnm_stage(self):
        self.assertNotEqual(protected_details('Prior 42 Gy in 14 fractions.'),
                            protected_details('Prior 14 Gy in 42 fractions.'))
        self.assertNotEqual(protected_details('Stage pT3N1a breast cancer.'),
                            protected_details('Stage pT1N3a breast cancer.'))
        self.assertEqual(protected_details('Stage pT3N1a breast cancer.'),
                         protected_details('Stage pT3 pN1a breast cancer.'))

    def test_added_need_rejected_by_checker_then_planned_fresh(self):
        query = NEW.replace('should be used?', 'should be used, and what toxicity is expected?')
        self.llm.generate_structured.side_effect = [{'equivalent': False, 'reason': 'Toxicity is a new need.'}, plan_payload()]
        result = self.planner.plan(self.analysis(query))
        self.assertEqual(result.planning['cache_status'], 'created')
        self.assertEqual(result.planning['llm_calls'], 2)
        self.assertEqual(result.planning['paraphrase_check_calls'], 1)
        self.assertEqual(len(list(Path(self.tmp.name).glob('*.json'))), 2)

    def test_bad_anchors_missing_units_and_nonboolean_confirmations_cannot_reuse(self):
        invalid = []
        proof = self.proof(); proof['anchors'][0]['anchors'][0]['quote'] = 'What dose'; invalid.append(proof)
        proof = self.proof(); proof['unit_mappings'].pop(); invalid.append(proof)
        proof = self.proof(); proof['facts_unchanged'] = 'true'; invalid.append(proof)
        proof = self.proof(); proof['unit_mappings'][0]['new_ids'] = ['C999']; invalid.append(proof)
        # Each failed verification uses a different wording so it cannot hit a previous fresh plan.
        for i, proof in enumerate(invalid):
            with self.subTest(i=i):
                query = NEW.replace('should be used?', 'should be used' + ' please' * (i+1) + '?')
                self.llm.generate_structured.side_effect = [proof, plan_payload()]
                result = self.planner.plan(self.analysis(query))
                self.assertEqual(result.planning['cache_status'], 'created')
                self.assertIn('validation failed', result.planning['paraphrase_miss_reason'])

    def test_unmapped_extra_current_sentence_cannot_be_silently_dropped(self):
        query = NEW.replace('She has never received RT.', 'She has never received RT. Surgery is scheduled.')
        self.llm.generate_structured.side_effect = [self.proof(), plan_payload()]
        result = self.planner.plan(self.analysis(query))
        self.assertEqual(result.planning['cache_status'], 'created')
        self.assertIn('Every current context unit', result.planning['paraphrase_miss_reason'])

    def test_model_setting_changes_do_not_use_semantic_candidates(self):
        self.cfg.llm.model_id = 'synthetic-other-model'
        self.llm.generate_structured.return_value = plan_payload()
        result = self.planner.plan(self.analysis(NEW))
        self.assertEqual(result.planning['cache_status'], 'created')
        self.assertEqual(result.planning['llm_calls'], 1)
        self.assertEqual(result.planning['paraphrase_check_calls'], 0)

    def test_aliases_are_bounded_and_do_not_create_new_plan_files(self):
        self.cfg.features.paraphrase_cache_aliases = 2
        self.cfg.features.plan_cache_path = str(Path(self.tmp.name) / 'bounded')
        self.planner = VersionedPlanner(self.llm, self.cfg)
        self.original = self.planner.plan(self.analysis(OLD))
        keys = []
        for word in ['preferred', 'recommended', 'appropriate']:
            query = NEW.replace('should be used', 'are ' + word)
            self.llm.generate_structured.return_value = self.proof(query)
            keys.append(self.planner.plan(self.analysis(query)).planning['cache_key'])
        paths = list(Path(self.cfg.features.plan_cache_path).glob('*.json'))
        self.assertEqual(len(paths), 1)
        aliases = json.loads(paths[0].read_text())['paraphrases']
        self.assertEqual([a['cache_key'] for a in aliases], keys[-2:])
        self.assertTrue(all('payload' not in a for a in aliases))

    def test_concurrent_paraphrases_do_not_lose_aliases(self):
        queries = [NEW, NEW.replace('should be used', 'are recommended')]
        def response(prompt, **kwargs):
            data = json.loads(prompt.split('\nDATA:\n', 1)[1])
            text = data['current']['sources']['question']
            return self.proof(NEW if text == input_sources(NEW)['question'] else queries[1])
        self.llm.generate_structured.side_effect = response
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda q: self.planner.plan(self.analysis(q)), queries))
        self.assertTrue(all(r.planning['cache_status'] == 'paraphrase_hit' for r in results))
        record = json.loads(next(Path(self.tmp.name).glob('*.json')).read_text())
        self.assertEqual(len(record['paraphrases']), 2)

    def test_concurrent_same_wording_verifies_only_once(self):
        self.llm.generate_structured.return_value = self.proof()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda q: self.planner.plan(self.analysis(q)), [NEW, NEW]))
        self.llm.generate_structured.assert_called_once()
        self.assertEqual({r.planning['cache_status'] for r in results}, {'paraphrase_hit', 'paraphrase_alias_hit'})

    def test_corrupted_alias_proof_is_rejected(self):
        self.llm.generate_structured.return_value = self.proof()
        self.planner.plan(self.analysis(NEW))
        path = next(Path(self.tmp.name).glob('*.json'))
        record = json.loads(path.read_text())
        record['paraphrases'][0]['verification']['reason'] = 'tampered'
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'verification is corrupted'):
            self.planner.plan(self.analysis(NEW))

    def test_cleanup_during_verification_replans_instead_of_reusing_a_deleted_source(self):
        def response(prompt, **kwargs):
            if 'Determine whether CURRENT' in prompt:
                limit = self.planner.cache.max_bytes
                self.planner.cache.max_bytes = 1
                self.planner.cache.maintain(self.planner.revision_identity())
                self.planner.cache.max_bytes = limit
                return self.proof()
            return plan_payload()
        self.llm.generate_structured.side_effect = response
        result = self.planner.plan(self.analysis(NEW))
        self.assertEqual(result.planning['cache_status'], 'created')
        self.assertEqual(result.planning['llm_calls'], 2)
        self.assertIn('evicted during verification', result.planning['paraphrase_miss_reason'])

    def test_alias_reads_refresh_the_canonical_plans_recency(self):
        self.llm.generate_structured.return_value = self.proof()
        self.planner.plan(self.analysis(NEW))
        path = next(Path(self.tmp.name).glob('*.json'))
        before = path.stat().st_mtime_ns
        self.planner.cache.clock = lambda: before/1_000_000_000 + 60
        self.planner.plan(self.analysis(NEW))
        self.assertGreater(path.stat().st_mtime_ns, before)

    def test_saved_alias_expires_with_its_canonical_plan(self):
        self.llm.generate_structured.return_value = self.proof()
        self.planner.plan(self.analysis(NEW))
        path = next(Path(self.tmp.name).glob('*.json'))
        record = json.loads(path.read_text())
        record['created_at'] = '2000-01-01T00:00:00+00:00'
        path.write_text(json.dumps(record))
        self.llm.reset_mock()
        self.llm.generate_structured.return_value = plan_payload()
        repeated = self.planner.plan(self.analysis(NEW))
        self.assertEqual(repeated.planning['cache_status'], 'created')
        self.assertEqual(repeated.planning['paraphrase_check_calls'], 0)
        self.llm.generate_structured.assert_called_once()

    def test_audit_can_be_retained_by_alias_key_and_survives_cache_cleanup(self):
        self.llm.generate_structured.return_value = self.proof()
        result = self.planner.plan(self.analysis(NEW))
        # Keep audits inside the test root, but outside its disposable cache directory.
        self.planner.cache.audit_folder = Path(self.tmp.name).parent/(Path(self.tmp.name).name + '_review')
        self.addCleanup(shutil.rmtree, self.planner.cache.audit_folder, ignore_errors=True)
        audit = self.planner.cache.retain(result.planning['cache_key'], 'Keep this wording for review')
        saved = json.loads(audit.read_text())
        self.assertEqual(saved['source_cache_key'], self.original.planning['cache_key'])
        self.assertEqual(saved['requested_key'], result.planning['cache_key'])
        self.assertEqual(saved['source_record']['paraphrases'][0]['verification'], self.proof())
        self.planner.cache.max_bytes = 1
        self.planner.cache.maintain(self.planner.revision_identity())
        self.assertTrue(audit.exists())

if __name__ == '__main__': unittest.main()
