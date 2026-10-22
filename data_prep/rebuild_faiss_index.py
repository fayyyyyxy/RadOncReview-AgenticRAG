"""Rebuild the review-note FAISS index from the current DynamoDB text, with every vector stored under its FAISS_ID.

Default run: back up the S3 index locally, rebuild, and check the FAISS_ID mapping; nothing in S3 changes.
--upload: copy the S3 index to a backup key, then replace it with the checked rebuild.
Prints IDs and counts only.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys

from botocore.config import Config as BotoConfig
import faiss
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config import Config, create_boto3_session

SAVED = Path(__file__).resolve().parent / "saved_data"
OUTPUT = SAVED / "rebuilt_index"


def scan_rows(table):
    """FAISS_ID -> text for every review-note row."""
    rows, kwargs = {}, {"ProjectionExpression": "FAISS_ID, #t", "ExpressionAttributeNames": {"#t": "text"}}
    while True:
        page = table.scan(**kwargs)
        rows.update({int(item["FAISS_ID"]): item["text"] for item in page["Items"]})
        if "LastEvaluatedKey" not in page:
            return rows
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def table_fingerprint(rows):
    """Changes whenever any row's ID or text changes."""
    return hashlib.sha256(json.dumps(sorted(rows.items())).encode("utf-8")).hexdigest()[:12]


def embed(bedrock, model_id, text):
    # Same request as ingestion and the query side, so documents and queries share one vector space
    response = bedrock.invoke_model(modelId=model_id, body=json.dumps({"inputText": text}))
    return json.loads(response["body"].read())["embedding"]


def build(rows, bedrock, model_id, workers=8):
    ids = np.array(sorted(rows), dtype="int64")
    vectors = []
    with ThreadPoolExecutor(workers) as pool:
        # 💡 map() returns results in input order, so vectors[k] belongs to ids[k] however the threads finish
        for n, vector in enumerate(pool.map(lambda i: embed(bedrock, model_id, rows[int(i)]), ids), 1):
            vectors.append(vector)
            if n % 1000 == 0:
                print(f"Embedded {n}/{len(ids)}")
    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    # 💡 IndexIDMap2 stores each vector's FAISS_ID, so a hit names its row directly instead of relying on
    # insertion order matching the table, which is how the January index drifted; "2" adds reconstruct()
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors, ids)
    return index


def stored_ids(index):
    """FAISS_IDs an index answers with: its ID map, or positions for a plain flat index."""
    if hasattr(index, "id_map"):
        return set(faiss.vector_to_array(index.id_map).tolist())
    return set(range(index.ntotal))


def end_to_end_misses(retriever, index, rows, sample):
    """Sampled rows whose own text, searched through the app's dense_search, does not come back as the top hit.

    The query is embedded fresh, so this fails if any vector sits under another row's ID.
    """
    retriever._faiss_index, retriever._faiss_index_loaded = index, True
    misses = []
    for i in sample:
        hits = retriever.dense_search(rows[i], top_k=1)
        # ⚠️ The table holds a duplicated document, so a row can come back as its twin; same text counts as found
        if not hits or hits[0].text != rows[i]:
            misses.append(i)
    return misses


def backup_path(s3, aws):
    head = s3.head_object(Bucket=aws.s3_bucket, Key=aws.faiss_index_key)
    stem = Path(aws.faiss_index_key).stem
    return SAVED / "backups" / f"{stem}_{head['LastModified']:%Y-%m-%d}.bin", head


def rebuild(config, session, sample_size, seed):
    from core.retriever import RetrieverManager
    aws = config.aws
    s3 = session.client("s3", region_name=aws.region)
    # 1. Back up the index currently in S3 (the bucket has no versioning, so an overwrite cannot be undone)
    backup, head = backup_path(s3, aws)
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(aws.s3_bucket, aws.faiss_index_key, str(backup))
    old = faiss.read_index(str(backup))
    if backup.stat().st_size != head["ContentLength"]:
        raise ValueError("Backup size differs from the S3 object.")

    # 2. Rebuild from the table, reusing an earlier rebuild of the same table
    rows = scan_rows(session.resource("dynamodb", region_name=aws.region).Table(aws.dynamodb_table))
    fingerprint = table_fingerprint(rows)
    index_path, manifest_path = OUTPUT / Path(aws.faiss_index_key).name, OUTPUT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["table_fingerprint"] == fingerprint:
        index = faiss.read_index(str(index_path))
    else:
        bedrock = session.client("bedrock-runtime", region_name=aws.region,
                                 config=BotoConfig(retries={"max_attempts": 10, "mode": "adaptive"}))
        index = build(rows, bedrock, config.embedding.model_id)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

    # 3. Check: same IDs as the table, and each sampled row's text finds that row, in the new index but not the old
    sample = sorted(random.Random(seed).sample(sorted(rows), sample_size))
    retriever = RetrieverManager(config.retriever, config.embedding, aws)
    new_misses = end_to_end_misses(retriever, index, rows, sample)
    old_misses = end_to_end_misses(retriever, old, rows, sample)
    checks = {
        "table_rows": len(rows), "vectors": index.ntotal, "same_ids_as_table": stored_ids(index) == set(rows),
        "sample": len(sample), "new_index_misses": new_misses,
        "old_index_found": len(sample) - len(old_misses),
    }
    passed = checks["same_ids_as_table"] and index.ntotal == len(rows) and not new_misses
    manifest = {
        "table_fingerprint": fingerprint, "embedding_model": config.embedding.model_id,
        "index_type": type(index).__name__, "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest()[:12],
        "backup": backup.name, "backup_vectors": old.ntotal, "checks": checks, "passed": passed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def upload(config, session):
    """Replace the S3 index with the checked rebuild, keeping the old one under a backup key."""
    aws = config.aws
    s3 = session.client("s3", region_name=aws.region)
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    if not manifest["passed"]:
        raise ValueError("The rebuild did not pass its checks.")
    # ⚠️ The table may have changed since the checks ran; the rebuild must still describe it
    rows = scan_rows(session.resource("dynamodb", region_name=aws.region).Table(aws.dynamodb_table))
    if table_fingerprint(rows) != manifest["table_fingerprint"]:
        raise ValueError("DynamoDB changed since the rebuild; run the rebuild again.")
    index_path = OUTPUT / Path(aws.faiss_index_key).name
    if hashlib.sha256(index_path.read_bytes()).hexdigest()[:12] != manifest["index_sha256"]:
        raise ValueError("The rebuilt index file changed since it was checked.")
    backup_key = f"backups/{manifest['backup']}"
    s3.copy_object(Bucket=aws.s3_bucket, Key=backup_key, CopySource={"Bucket": aws.s3_bucket, "Key": aws.faiss_index_key})
    s3.upload_file(str(index_path), aws.s3_bucket, aws.faiss_index_key)
    if s3.head_object(Bucket=aws.s3_bucket, Key=aws.faiss_index_key)["ContentLength"] != index_path.stat().st_size:
        raise ValueError("Uploaded size differs from the rebuilt index.")
    return {"uploaded": aws.faiss_index_key, "previous_index_at": backup_key, "index_sha256": manifest["index_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="Replace the S3 index with the checked rebuild.")
    parser.add_argument("--sample", type=int, default=100, help="Rows checked end to end.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        config = Config()
        session = create_boto3_session(config.aws)
        result = upload(config, session) if args.upload else rebuild(config, session, args.sample, args.seed)
        print(json.dumps(result, indent=2))
        return 0 if result.get("passed", True) else 1
    except Exception as exc:
        print(f"FAISS rebuild failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
