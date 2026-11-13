"""
Case Labels: the four labels a user selects with a question, in the scenario workbook's vocabulary.
They route evidence and select past cases; they are never added to the answer-writing prompt.
"""
from dataclasses import dataclass


NOT_SPECIFIED = "Not specified"

DISEASE_SITES = (
    "Breast", "CNS", "Gastrointestinal", "Genitourinary (incl. Prostate)", "Gynecologic", "Head & Neck",
    "Lung & Thoracic", "Lymphoma & Hematologic", "Pediatrics", "Sarcoma", "Skin", "Benign & HO Prophylaxis",
    "Oligometastatic", "Palliative",
)

# Workbook Setting value -> setting group offered to users (needs clinician review)
SETTING_GROUPS = {
    "Definitive": "Definitive", "Definitive (local control)": "Definitive",
    "Definitive / adjuvant": "Definitive", "Consolidative": "Definitive",
    "Adjuvant": "Adjuvant", "Adjuvant / salvage": "Adjuvant",
    "Neoadjuvant": "Neoadjuvant", "Neoadjuvant / definitive": "Neoadjuvant",
    "Salvage": "Salvage", "Salvage / residual": "Salvage",
    "Palliative": "Palliative",
    "Metastasis-directed": "Metastasis-directed",
    "Benign": "Benign",
    "Skin-directed": "Skin-directed", "Skin-directed (trial)": "Skin-directed",
    "Conditioning (TBI)": "Other", "Prophylactic (PCI)": "Other",
}

PRIOR_RT = (
    "None", "Prior RT – different site", "Prior RT – adjacent", "Prior RT – cumulative OAR dose",
    "Prior RT – incomplete course", "Prior RT – same course (cycle)", "Re-irradiation (overlap)",
    "Re-irradiation (adjacent)", "Re-irradiation (cumulative)", "Re-irradiation (remote)",
)

ENCOUNTER_TYPES = ("New start", "Boost", "Brachytherapy", "Replan", "Resimulation", "Sim & treat", "Upcoming start")


def prior_rt_group(value: str) -> str:
    """Past cases match on whether there was any earlier RT; ten values would split small sites too finely."""
    if value == NOT_SPECIFIED:
        return NOT_SPECIFIED
    return "No prior RT" if value == "None" else "Prior RT"

CHOICES = {
    "disease_site": DISEASE_SITES,
    "setting_group": tuple(dict.fromkeys(SETTING_GROUPS.values())),
    "prior_rt": PRIOR_RT,
    "encounter_type": ENCOUNTER_TYPES,
}


@dataclass(frozen=True)
class CaseLabels:
    """User-selected labels; any of them may be Not specified."""
    disease_site: str = NOT_SPECIFIED
    setting_group: str = NOT_SPECIFIED
    prior_rt: str = NOT_SPECIFIED
    encounter_type: str = NOT_SPECIFIED

    def __post_init__(self):
        # ⚠️ Reject near misses ("Lung" for "Lung & Thoracic"): a wrong site would filter out the right evidence
        for name, allowed in CHOICES.items():
            value = getattr(self, name)
            if value != NOT_SPECIFIED and value not in allowed:
                raise ValueError(f"{name} must be one of the listed choices or {NOT_SPECIFIED!r}.")
