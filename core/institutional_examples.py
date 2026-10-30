"""Adapter for adding institutional examples to synthesis, separate from evidence."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .case_labels import CaseLabels, NOT_SPECIFIED, SETTING_GROUPS, prior_rt_group
from .past_cases import LABEL, MIN_SIMILARITY, NumpyPastCaseIndex, search_text, select_examples


@dataclass(frozen=True)
class CaseContext:
    """Structured input and study linkage. Linkage is never embedded or put in the prompt."""
    vignette: str
    clinical_question: str
    case_id: str
    patient_id: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in vars(self).values()):
            raise ValueError("Institutional examples require vignette, question, case ID and patient ID.")

    def query(self) -> str:
        return f"Patient vignette:\n{self.vignette}\n\nClinical question:\n{self.clinical_question}"


class LocalInstitutionalExamples:
    """Reads the existing local index format; does not build indexes or call AWS itself.

    A different storage implementation can supply the same select(context, labels, embed)
    and provenance()/validate_context() methods. The engine need not know its storage format.
    """

    def __init__(self, folder, embedding_model: str, dimension: int,
                 expected_source_sha256: Optional[str] = None):
        folder = Path(folder)
        try:
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            source_bytes = (folder.parent / "development_examples.jsonl").read_bytes()
            self.index = NumpyPastCaseIndex.load(folder)
        except FileNotFoundError as exc:
            raise ValueError("Institutional example index or its development export is missing; "
                             "supply the existing index before enabling examples.") from exc
        if manifest.get("embedding_model") != embedding_model or manifest.get("dimension") != dimension:
            raise ValueError("Institutional example index uses a different embedding model or dimension.")
        if self.index.dimension != dimension or not np.isfinite(self.index.vectors).all():
            raise ValueError("Institutional example vectors have invalid dimensions or values.")
        if manifest.get("source_sha256") != hashlib.sha256(source_bytes).hexdigest():
            raise ValueError("Institutional example index does not match the current development export.")
        if expected_source_sha256 is not None and manifest["source_sha256"] != expected_source_sha256:
            raise ValueError("Institutional example index was built from a different workbook development corpus.")

        examples = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
        approved = {}
        for example in examples:
            governance = example["governance"]
            if governance["split"] != "Development" or governance["worked_example_eligible"] != "Yes":
                raise ValueError("Only eligible Development cases may enter the institutional example index.")
            if example["case_id"] in approved:
                raise ValueError("Duplicate case in development export.")
            approved[example["case_id"]] = example
        indexed_ids = [record["case_id"] for record in self.index.records]
        if len(set(indexed_ids)) != len(indexed_ids) or set(indexed_ids) != set(approved):
            raise ValueError("Indexed case IDs do not match the eligible development cases.")
        for record in self.index.records:
            example = approved[record["case_id"]]
            meta = example["metadata"]
            expected = {
                "patient_id": example["patient_linkage"]["patient_id"],
                "vignette": example["vignette"], "clinical_question": example["clinical_question"],
                "attending_plan": example["attending_plan"], "disease_site": meta["disease_site"],
                "setting_group": SETTING_GROUPS[meta["setting"]],
                "prior_rt_group": prior_rt_group(meta["prior_rt"]), "encounter_type": meta["encounter_type"],
                "label": LABEL,
                "search_sha256": hashlib.sha256(search_text(example["vignette"],
                                                           example["clinical_question"]).encode("utf-8")).hexdigest(),
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError("Institutional index payload differs from the development export.")
        self.version = self.index.version()
        if manifest.get("version") != self.version or manifest.get("cases") != len(indexed_ids):
            raise ValueError("Institutional index contents differ from its manifest.")
        self.embedding_model = embedding_model
        self.source_sha256 = manifest["source_sha256"]
        self.records = {r["case_id"]: r for r in self.index.records}
        self.positions = {r["case_id"]: i for i, r in enumerate(self.index.records)}

    def provenance(self) -> Dict:
        return {"index_version": self.version, "embedding_model": self.embedding_model,
                "source_sha256": self.source_sha256, "min_similarity": MIN_SIMILARITY, "max_examples": 2}

    def validate_context(self, context: CaseContext) -> None:
        record = self.records.get(context.case_id)
        if record is not None and any(record[key] != getattr(context, key)
                                      for key in ("vignette", "clinical_question", "patient_id")):
            raise ValueError("Query linkage or text does not match the indexed development case.")

    def select(self, context: CaseContext, labels: CaseLabels, embed: Callable) -> List[Dict]:
        self.validate_context(context)
        if labels.disease_site == NOT_SPECIFIED:
            return []
        # A validated indexed development query can reuse its existing embedding.
        if context.case_id in self.positions:
            vector = self.index.vectors[self.positions[context.case_id]]
        else:
            vector = embed(search_text(context.vignette, context.clinical_question))
        hits = select_examples(self.index, vector, labels, context.case_id, context.patient_id,
                               k=2, min_similarity=MIN_SIMILARITY)
        # Study IDs support auditing; patient IDs and delivered-treatment metadata stay out of context.
        return [{"case_id": hit.record["case_id"], "similarity": hit.similarity, "label": LABEL,
                 "vignette": hit.record["vignette"], "clinical_question": hit.record["clinical_question"],
                 "attending_plan": hit.record["attending_plan"]} for hit in hits]
