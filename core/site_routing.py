"""
Site Routing: the evidence categories that belong to each user-selected disease site.
The table needs clinician review. A source with no category for a site contributes nothing for that site.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional

from .case_labels import DISEASE_SITES, NOT_SPECIFIED


# Site -> (review-note `category` values in DynamoDB, trial-paper folders under reading-lists/ in S3);
# None searches every category
SITE_EVIDENCE = {
    "Breast": (("Breast",), ("Breast",)),
    # Brain metastases are CNS cases in the workbook, and their notes sit with oligometastatic disease
    "CNS": (("CNS|Pediatrics", "BrainMets|Oligo|Immuno"), ("CNS",)),
    "Gastrointestinal": (("Gastrointestinal",), ("Gastrointestinal",)),
    "Genitourinary (incl. Prostate)": (("Genitourinary",), ("Genitourinary", "Prostate")),
    "Gynecologic": (("Gynecology",), ("Gynecologic",)),
    "Head & Neck": (("Head&Neck|Skin",), ("Head & Neck",)),
    "Lung & Thoracic": (("Thorax",), ("Lung",)),
    "Lymphoma & Hematologic": (("Hematologic",), ("Lymphoma",)),
    "Pediatrics": (("CNS|Pediatrics",), ("Pediatrics",)),
    "Sarcoma": (("Sarcoma",), ("Sarcoma",)),
    "Skin": (("Head&Neck|Skin",), ("Skin",)),
    "Oligometastatic": (("BrainMets|Oligo|Immuno",), ("Oligometastatic",)),
    # No review-note document covers these sites, so they draw on their trial-paper folders only
    "Palliative": ((), ("Palliative",)),
    "Benign & HO Prophylaxis": ((), ("HO Prophylaxis",)),
}
if set(SITE_EVIDENCE) != set(DISEASE_SITES):
    raise ValueError("SITE_EVIDENCE must list every disease site exactly once.")


@dataclass(frozen=True)
class EvidenceScope:
    """Evidence a query may draw on. None leaves a source unfiltered; an empty set excludes it."""
    review_note_categories: Optional[FrozenSet[str]] = None
    trial_paper_folders: Optional[FrozenSet[str]] = None

    def allows(self, chunk: Dict) -> bool:
        """Whether a retrieved chunk falls inside the scope; KB chunks carry their folder as category."""
        allowed = (self.trial_paper_folders if chunk.get("source_type") == "clinical_trial"
                   else self.review_note_categories)
        return allowed is None or chunk.get("category") in allowed


def scope_for(site: str) -> EvidenceScope:
    if site == NOT_SPECIFIED:
        return EvidenceScope()
    notes, folders = SITE_EVIDENCE[site]
    return EvidenceScope(None if notes is None else frozenset(notes), frozenset(folders))


def table_fingerprint() -> str:
    """Changes whenever the hand-edited table changes; recorded with every run."""
    return hashlib.sha256(json.dumps(SITE_EVIDENCE, sort_keys=True).encode("utf-8")).hexdigest()[:12]
