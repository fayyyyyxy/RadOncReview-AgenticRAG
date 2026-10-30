"""Inspect/prune disposable plans, or explicitly retain a separate audit snapshot. No model calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from core.versioned_planner import VersionedPlanner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-cache', type=Path)
    parser.add_argument('--audit-path', type=Path)
    parser.add_argument('--max-bytes', type=int)
    parser.add_argument('--ttl-days', type=float)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prune', action='store_true', help='Apply expiry and disk-budget eviction now.')
    action.add_argument('--retain', metavar='CACHE_KEY', help='Copy this plan to the separate audit folder before cleanup.')
    parser.add_argument('--note', default='')
    args = parser.parse_args(argv)
    if args.note and not args.retain:
        parser.error('--note requires --retain')
    config = Config()
    if args.plan_cache is not None:
        config.features.plan_cache_path = str(args.plan_cache)
    if args.audit_path is not None:
        config.features.plan_audit_path = str(args.audit_path)
    if args.max_bytes is not None:
        config.features.plan_cache_max_bytes = args.max_bytes
    if args.ttl_days is not None:
        config.features.plan_cache_ttl_days = args.ttl_days
    planner = VersionedPlanner(None, config)
    if args.retain:
        result = {'audit_snapshot': str(planner.cache.retain(args.retain, args.note)), 'aws_calls': 0}
    elif args.prune:
        result = planner.cache.maintain(planner.revision_identity())
    else:
        result = planner.cache.inspect(planner.revision_identity())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
