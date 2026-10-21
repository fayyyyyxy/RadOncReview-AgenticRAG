"""Build or update the local past-case index from development cases; dry-run by default.

Only development cases eligible as worked examples are indexed. The search key is the vignette plus the clinical
question (workbook step E7); the attending plan and labels are payload. Only new or reworded cases are embedded;
cases whose labels alone changed keep their vectors.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from core.case_labels import SETTING_GROUPS, prior_rt_group
from core.past_cases import LABEL, NumpyPastCaseIndex

PROCESSED = ROOT / "data" / "processed"
INDEX_FOLDER = PROCESSED / "past_case_index"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_record(example):
    """Index record for one development case; refuses anything that may not be shown as an example."""
    governance = example["governance"]
    if governance["split"] != "Development" or governance["worked_example_eligible"] != "Yes":
        raise ValueError(f"{example['case_id']} is not an eligible development case.")
    meta = example["metadata"]
    if meta["setting"] not in SETTING_GROUPS:
        raise ValueError(f"{example['case_id']}: Setting value has no setting group.")
    return {
        "case_id": example["case_id"],
        "patient_id": example["patient_linkage"]["patient_id"],
        "disease_site": meta["disease_site"],
        "setting_group": SETTING_GROUPS[meta["setting"]],
        "encounter_type": meta["encounter_type"],
        "setting": meta["setting"],
        "prior_rt": meta["prior_rt"],
        "prior_rt_group": prior_rt_group(meta["prior_rt"]),
        "technique": meta["technique"],
        "subsite": meta["subsite"],
        "vignette": example["vignette"],
        "clinical_question": example["clinical_question"],
        "attending_plan": example["attending_plan"],
        "label": LABEL,
        "search_sha256": sha256_text(example["search_text"]),
    }


def stale(records, index):
    """Records that are new or whose search text changed since the last build."""
    known = {r["case_id"]: r["search_sha256"] for r in index.records}
    return [r for r in records if known.get(r["case_id"]) != r["search_sha256"]]


def relabelled(records, index):
    """Records with unchanged search text but changed labels or payload, with the vectors they already have."""
    known = {r["case_id"]: i for i, r in enumerate(index.records)}
    # ⚠️ A label fix in the workbook (Review Changes) leaves the text alone; checking text only would miss it
    changed = [r for r in records if r["case_id"] in known
               and index.records[known[r["case_id"]]]["search_sha256"] == r["search_sha256"]
               and index.records[known[r["case_id"]]] != r]
    return changed, index.vectors[[known[r["case_id"]] for r in changed]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Embed new or changed cases with Bedrock.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        lines = (PROCESSED / "development_examples.jsonl").read_text(encoding="utf-8").splitlines()
        examples = {e["case_id"]: e for e in map(json.loads, filter(str.strip, lines))}
        records = [to_record(e) for e in examples.values()]
        index = NumpyPastCaseIndex.load(INDEX_FOLDER) if (INDEX_FOLDER / "past_cases.npy").exists() else None
        todo = stale(records, index) if index else records
        relabel, kept_vectors = relabelled(records, index) if index else ([], None)
        removed = sorted({r["case_id"] for r in index.records} - set(examples)) if index else []
        if removed:
            raise ValueError(f"{len(removed)} indexed cases are no longer development examples; rebuild from scratch.")
        if not args.execute:
            print(json.dumps({"mode": "dry_run", "model_calls": 0, "development_cases": len(records),
                              "indexed": len(index.records) if index else 0, "to_embed": len(todo),
                              "to_relabel": len(relabel)}, indent=2))
            return 0

        from config import Config, create_boto3_session
        config = Config()
        client = create_boto3_session(config.aws).client("bedrock-runtime", region_name=config.aws.region)

        def embed(record):
            text = examples[record["case_id"]]["search_text"]
            response = client.invoke_model(modelId=config.embedding.model_id, body=json.dumps({"inputText": text}))
            return json.loads(response["body"].read())["embedding"]

        index = index or NumpyPastCaseIndex(config.embedding.dimension)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            vectors = list(pool.map(embed, todo))
        if todo:
            index.add(todo, vectors)
        if relabel:
            index.add(relabel, kept_vectors)
        index.save(INDEX_FOLDER)
        manifest = {
            "version": index.version(), "cases": len(index.records), "embedded_now": len(todo),
            "relabelled_now": len(relabel),
            "embedding_model": config.embedding.model_id, "dimension": index.dimension,
            "source_sha256": hashlib.sha256((PROCESSED / "development_examples.jsonl").read_bytes()).hexdigest(),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (INDEX_FOLDER / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0
    except Exception as exc:
        print(f"Index build failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
