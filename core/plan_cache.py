"""Bounded disposable plan storage and explicitly retained, separate audit snapshots."""
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

from .versioned_planner import cache_lock

KEY = re.compile(r'[0-9a-f]{64}')


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                      default=lambda x: x.value if isinstance(x, Enum) else str(x)).encode('utf-8')


def revision(identity):
    """Planner/prompt revisions, independent of case, model settings and cache capacity."""
    identity = identity if isinstance(identity, dict) else {}
    fields = {k: identity.get(k) for k in ('version', 'code_sha256', 'llm_client_sha256', 'prompt_sha256')}
    policy = identity.get('paraphrase_policy') or {}
    policy = policy if isinstance(policy, dict) else {}
    fields['policy'] = {k: policy.get(k) for k in ('version', 'code_sha256', 'prompt_sha256', 'analysis_code_sha256')}
    return hashlib.sha256(encode(fields)).hexdigest()


def atomic_bytes(path, data):
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with os.fdopen(os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'wb') as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PlanCache:
    def __init__(self, folder, max_bytes=100_000_000, ttl_days=30, audit_folder=None, clock=None):
        self.folder = Path(folder)
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError('Plan cache byte budget must be a positive integer')
        if isinstance(ttl_days, bool) or not isinstance(ttl_days, (int, float)) or not math.isfinite(ttl_days) or ttl_days <= 0:
            raise ValueError('Plan cache lifetime must be positive and finite')
        self.max_bytes, self.ttl_seconds = max_bytes, ttl_days * 86400
        self.audit_folder = Path(audit_folder) if audit_folder else self.folder.with_name(self.folder.name + '_audits')
        self._validate_audit_path()
        self.clock = clock
        self.index_path = self.folder / '.cache-index'

    def now(self):
        return self.clock() if self.clock else time.time()

    def _validate_audit_path(self):
        cache, audit = self.folder.resolve(), self.audit_folder.resolve()
        if cache == audit or cache in audit.parents or audit in cache.parents:
            raise ValueError('Audit and disposable cache folders must be separate, non-overlapping directories')

    def _path(self, key):
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise ValueError('A plan cache key must be a SHA-256 digest')
        return self.folder / (key + '.json')

    def _created(self, record, fallback):
        try:
            created = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00'))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            created = created.timestamp()
            if not math.isfinite(created):
                raise ValueError('Invalid creation time')
        except (ValueError, KeyError, TypeError, AttributeError):
            created = fallback
        return created

    def _metadata(self, stat, record):
        created = self._created(record, stat.st_mtime)
        return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'created': created,
                'last_used': stat.st_mtime, 'revision': revision(record.get('identity', {}))}

    def _inventory(self, clean=False):
        if clean:
            # Every current cache writer holds cache-storage, so these are abandoned writes.
            temporary = re.compile(r'(?:\.[0-9a-f]{64}\.json|\.\.cache-index)\.[0-9a-f]{32}\.tmp')
            for path in self.folder.glob('.*.tmp'):
                if temporary.fullmatch(path.name) and not path.is_symlink():
                    path.unlink(missing_ok=True)
        try:
            saved = json.loads(self.index_path.read_text())
            saved = saved['entries'] if saved.get('schema') == 1 else {}
            if not isinstance(saved, dict):
                saved = {}
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            saved = {}
        entries = {}
        for path in self.folder.glob('*.json'):
            if path.is_symlink() or not KEY.fullmatch(path.stem) or not path.is_file():
                continue
            stat = path.stat()
            old = saved.get(path.stem, {})
            if isinstance(old, dict) and old.get('size') == stat.st_size and old.get('mtime_ns') == stat.st_mtime_ns and all(
                isinstance(old.get(k), (int, float)) and math.isfinite(old[k]) for k in ('created', 'last_used')):
                entries[path.stem] = old
            else:
                try:
                    record = json.loads(path.read_text())
                    if not isinstance(record, dict):
                        record = {}
                except (OSError, ValueError):
                    record = {}
                entries[path.stem] = self._metadata(stat, record)
        return entries

    def _index_bytes(self, entries):
        return encode({'schema': 1, 'entries': entries}) if entries else b''

    def _usage(self, entries):
        return sum(r['size'] for r in entries.values()) + len(self._index_bytes(entries))

    def _flush(self, entries):
        if entries:
            atomic_bytes(self.index_path, self._index_bytes(entries))
        else:
            self.index_path.unlink(missing_ok=True)

    def _prune(self, entries, identity):
        now, current = self.now(), revision(identity)
        expired = [key for key, row in entries.items() if now - row['created'] >= self.ttl_seconds]
        for key in expired:
            self._path(key).unlink(missing_ok=True)
            del entries[key]
        evicted, obsolete = 0, 0
        used = self._usage(entries)
        order = sorted(entries, key=lambda key: (entries[key]['revision'] == current, entries[key]['last_used'], key))
        for key in order:
            if used <= self.max_bytes:
                break
            obsolete += entries[key]['revision'] != current
            removed = entries[key]['size'] + len(encode(key)) + 1 + len(encode(entries[key])) + 1
            self._path(key).unlink(missing_ok=True)
            del entries[key]
            used = used-removed if entries else 0
            evicted += 1
        return {'expired': len(expired), 'evicted': evicted, 'obsolete_evicted': obsolete,
                'bytes_used': self._usage(entries), 'max_bytes': self.max_bytes, 'plans': len(entries)}

    def _touch(self, key, entries):
        now = self.now()
        path = self._path(key)
        os.utime(path, ns=(path.stat().st_atime_ns, int(now * 1_000_000_000)))
        entries[key].update(last_used=now, mtime_ns=path.stat().st_mtime_ns)

    def inspect(self, identity):
        """Read-only preview, including expired bytes; never expires a retained audit."""
        with cache_lock(self.folder, 'cache-storage'):
            entries = self._inventory()
            now, current = self.now(), revision(identity)
            unmanaged = [p for p in self.folder.iterdir() if p.is_file() and not p.is_symlink()
                         and p != self.index_path and not p.name.endswith('.lock') and p.stem not in entries]
            actual_bytes = sum(r['size'] for r in entries.values()) + (self.index_path.stat().st_size if self.index_path.exists() else 0)
            return {'bytes_used': actual_bytes, 'max_bytes': self.max_bytes, 'plans': len(entries),
                    'expired_plans': sum(now-r['created'] >= self.ttl_seconds for r in entries.values()),
                    'old_revision_plans': sum(r['revision'] != current for r in entries.values()),
                    'ttl_days': self.ttl_seconds/86400, 'audit_folder': str(self.audit_folder),
                    'unmanaged_bytes': sum(p.stat().st_size for p in unmanaged), 'unmanaged_files': len(unmanaged)}

    def maintain(self, identity):
        with cache_lock(self.folder, 'cache-storage'):
            entries = self._inventory(clean=True)
            report = self._prune(entries, identity)
            self._flush(entries)
            return report

    def load(self, key, identity):
        with cache_lock(self.folder, 'cache-storage'):
            entries = self._inventory(clean=True)
            row = entries.get(key)
            if row and self.now()-row['created'] < self.ttl_seconds:
                self._touch(key, entries)
            self._prune(entries, identity)
            record = json.loads(self._path(key).read_text()) if key in entries else None
            self._flush(entries)
            return record

    def candidates(self, identity, limit=128):
        with cache_lock(self.folder, 'cache-storage'):
            entries = self._inventory(clean=True)
            self._prune(entries, identity)
            records = []
            for key in sorted(entries, key=lambda k: entries[k]['last_used'], reverse=True)[:limit]:
                try:
                    record = json.loads(self._path(key).read_text())
                    if isinstance(record, dict) and record.get('cache_key') == key:
                        records.append(record)
                except (ValueError, OSError):
                    continue
            self._flush(entries)
            return records

    def touch(self, key, identity):
        return self.load(key, identity) is not None

    def _save(self, key, record, entries, identity):
        data = encode(record)
        path = self._path(key)
        now = self.now()
        metadata = {'size': len(data), 'mtime_ns': int(now*1_000_000_000),
                    'created': self._created(record, now), 'last_used': now,
                    'revision': revision(record.get('identity', {}))}
        # Include index overhead in admission; an oversized plan remains usable in memory.
        if len(data) + len(self._index_bytes({key: metadata})) > self.max_bytes:
            return {'stored': False, 'reason': 'record_exceeds_budget', 'available': key in entries}
        if self.now()-metadata['created'] >= self.ttl_seconds:
            return {'stored': False, 'reason': 'record_expired', 'available': key in entries}
        atomic_bytes(path, data)
        entries[key] = self._metadata(path.stat(), record)
        self._touch(key, entries)
        result = self._prune(entries, identity)
        return {**result, 'stored': key in entries, 'available': key in entries,
                'reason': None if key in entries else 'budget_eviction'}

    def store(self, key, record, identity):
        with cache_lock(self.folder, 'cache-storage'):
            entries = self._inventory(clean=True)
            self._prune(entries, identity)
            result = self._save(key, record, entries, identity)
            self._flush(entries)
            return result

    def update(self, key, identity, transform):
        with cache_lock(self.folder, 'cache-storage'):
            entries = self._inventory(clean=True)
            self._prune(entries, identity)
            if key not in entries:
                self._flush(entries)
                return {'stored': False, 'available': False, 'reason': 'source_expired_or_evicted'}
            record = transform(json.loads(self._path(key).read_text()))
            result = self._save(key, record, entries, identity)
            if not result['stored'] and result['available']:
                self._touch(key, entries)
                self._prune(entries, identity)
                result['available'] = key in entries
            self._flush(entries)
            return result

    def retain(self, key, note='', request_plan=None):
        """Explicit archival only. The audit folder is outside all cache cleanup."""
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError('An audit note must be text of at most 2000 characters')
        self._validate_audit_path()
        with cache_lock(self.folder, 'cache-storage'):
            path = self._path(key)
            if path.is_symlink():
                raise ValueError('Cannot archive a symbolic link')
            record = json.loads(path.read_text()) if path.exists() else None
            source_key = key
            if record is None:
                for candidate in self.folder.glob('*.json'):
                    if candidate.is_symlink() or not KEY.fullmatch(candidate.stem):
                        continue
                    try:
                        saved = json.loads(candidate.read_text())
                        aliases = saved.get('paraphrases', [])
                        if any(a.get('cache_key') == key for a in aliases):
                            record, source_key = saved, candidate.stem
                            break
                    except (OSError, ValueError, AttributeError, TypeError):
                        continue
            if record is None and request_plan is None:
                raise ValueError('The cache record no longer exists; retain a saved TaskPlan snapshot instead')
            snapshot = {'audit_version': 'plan_audit_v1', 'source_cache_key': source_key, 'requested_key': key, 'note': note,
                        'retained_at': datetime.fromtimestamp(self.now(), timezone.utc).isoformat(),
                        'source_record': record, 'request_plan': request_plan}
            self.audit_folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            audit = self.audit_folder / (key + '-' + uuid.uuid4().hex + '.json')
            atomic_bytes(audit, encode(snapshot))
            return audit
