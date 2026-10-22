"""
Past Cases: an index of earlier institutional cases and the rule that picks examples for a new case.
Examples show institutional practice; they are never evidence and never cited.
"""
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from .case_labels import NOT_SPECIFIED, CaseLabels, prior_rt_group


LABEL = "Institutional example – not evidence"
# Set on development cases (patient_cases/check_past_case_quality.py): 0.50 drops examples for 8 of 363 queries and
# keeps 75.5% of them with an example of the same technique (76.2% with no floor); 0.55 would drop 26
MIN_SIMILARITY = 0.50
# Every record carries these; other fields are payload shown with the example
KEYS = ("case_id", "patient_id", "disease_site", "setting_group", "prior_rt_group", "encounter_type")


@dataclass
class PastCaseHit:
    record: Dict
    similarity: float


class PastCaseIndex(ABC):
    """Storage-independent past-case search; a local and a cloud index share this contract."""

    @abstractmethod
    def add(self, records: List[Dict], vectors: np.ndarray) -> None:
        """Insert records, replacing any with the same case_id."""

    @abstractmethod
    def search(self, query: np.ndarray, filters: Dict[str, str], exclude: Dict[str, Set[str]],
               k: int) -> List[PastCaseHit]:
        """The k most similar records whose fields equal `filters` and whose values avoid `exclude`."""

    @abstractmethod
    def version(self) -> str:
        """An identifier that changes whenever the indexed content changes."""


def search_text(vignette: str, clinical_question: str) -> str:
    """The text a case is embedded by; a new case must be embedded the same way as the indexed ones."""
    return f"{vignette}\n\n{clinical_question}"


def normalise(vectors) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class NumpyPastCaseIndex(PastCaseIndex):
    """Exact cosine search over every row, kept as a .npy matrix plus a .jsonl file in the same order."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.records: List[Dict] = []
        self.vectors = np.zeros((0, dimension), dtype=np.float32)

    def add(self, records, vectors):
        vectors = normalise(vectors).reshape(len(records), -1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vectors must have {self.dimension} dimensions.")
        position = {r["case_id"]: i for i, r in enumerate(self.records)}
        new_records, new_vectors = [], []
        for record, vector in zip(records, vectors):
            missing = [key for key in KEYS if key not in record]
            if missing:
                raise ValueError(f"Record is missing {', '.join(missing)}.")
            if record["case_id"] in position:
                self.records[position[record["case_id"]]] = record
                self.vectors[position[record["case_id"]]] = vector
            else:
                position[record["case_id"]] = len(self.records) + len(new_records)
                new_records.append(record)
                new_vectors.append(vector)
        if new_records:
            self.records += new_records
            self.vectors = np.vstack([self.vectors, np.stack(new_vectors)])

    def search(self, query, filters, exclude, k):
        keep = np.array([
            all(r[field] == value for field, value in filters.items())
            and not any(r[field] in values for field, values in exclude.items())
            for r in self.records
        ], dtype=bool)
        if not keep.any():
            return []
        # 💡 Rows are unit length, so one matrix-vector product gives every cosine similarity at once
        scores = np.where(keep, self.vectors @ normalise(query), -np.inf)
        order = np.argsort(-scores, kind="stable")[:min(k, int(keep.sum()))]
        return [PastCaseHit(self.records[i], float(scores[i])) for i in order]

    def version(self):
        digest = hashlib.sha256()
        for i in sorted(range(len(self.records)), key=lambda i: self.records[i]["case_id"]):
            digest.update(json.dumps(self.records[i], sort_keys=True, ensure_ascii=False).encode("utf-8"))
            digest.update(self.vectors[i].tobytes())
        return digest.hexdigest()[:12]

    def save(self, folder) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "past_cases.npy", self.vectors)
        with (folder / "past_cases.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, folder) -> "NumpyPastCaseIndex":
        folder = Path(folder)
        vectors = np.load(folder / "past_cases.npy")
        lines = (folder / "past_cases.jsonl").read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        if len(records) != len(vectors):
            raise ValueError("past_cases.jsonl and past_cases.npy have different row counts.")
        index = cls(vectors.shape[1])
        index.records, index.vectors = records, vectors
        return index


def select_examples(index: PastCaseIndex, query: np.ndarray, labels: CaseLabels, case_id: Optional[str] = None,
                    patient_id: Optional[str] = None, k: int = 2, window: int = 10,
                    min_similarity: Optional[float] = None) -> List[PastCaseHit]:
    """Same site, setting group and prior-RT group first; then drop prior RT, then setting; never another site.

    A Not specified label is left out of every tier. Within each tier the `window` most similar records are
    re-ranked with the same encounter type first. case_id and patient_id exclude the case itself and the patient's
    other encounters (evaluation only). min_similarity, when set, drops weaker examples in every tier, so a case
    may get fewer than k examples or none.
    """
    if labels.disease_site == NOT_SPECIFIED:
        return []
    # Least important condition last, so each looser tier drops one condition from the end
    conditions = [(field, value) for field, value in (("setting_group", labels.setting_group),
                                                      ("prior_rt_group", prior_rt_group(labels.prior_rt)))
                  if value != NOT_SPECIFIED]
    tiers = [{"disease_site": labels.disease_site, **dict(conditions[:n])} for n in range(len(conditions), -1, -1)]
    chosen: List[PastCaseHit] = []
    for filters in tiers:
        # ⚠️ The looser tier would return the cases the stricter tier already chose; exclude them
        exclude = {"case_id": {case_id} - {None} | {h.record["case_id"] for h in chosen},
                   "patient_id": {patient_id} - {None}}
        hits = index.search(query, filters, exclude, window)
        if min_similarity is not None:
            hits = [h for h in hits if h.similarity >= min_similarity]
        hits.sort(key=lambda h: (h.record["encounter_type"] != labels.encounter_type, -h.similarity))
        chosen += hits[:k - len(chosen)]
        if len(chosen) == k:
            break
    return chosen
