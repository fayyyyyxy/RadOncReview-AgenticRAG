"""Stable passage identities for ranking and auditing, independent of result order."""
import hashlib


def passage_identity(chunk):
    get = chunk.get if isinstance(chunk, dict) else lambda key, default=None: getattr(chunk, key, default)
    source_type = get("source_type", "review_notes")
    source = (get("s3_location") or get("source_path") or get("path") or "") if source_type == "clinical_trial" else str(get("faiss_id", ""))
    digest = hashlib.sha256((get("text", "") or "").encode("utf-8")).hexdigest()
    return source_type, source, digest
