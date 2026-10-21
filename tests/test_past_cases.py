"""Case labels, the local past-case index and example selection, without AWS."""

import tempfile
import unittest

import numpy as np

from core.case_labels import NOT_SPECIFIED, CaseLabels, prior_rt_group
from core.past_cases import NumpyPastCaseIndex, select_examples


def record(case_id, patient_id, site="Breast", setting="Adjuvant", encounter="New start", prior="No prior RT"):
    return {"case_id": case_id, "patient_id": patient_id, "disease_site": site, "setting_group": setting,
            "prior_rt_group": prior, "encounter_type": encounter, "attending_plan": f"Plan of {case_id}"}


def build(rows):
    """rows: (record, vector) pairs; vectors are 3-D so closeness to the query is easy to control."""
    index = NumpyPastCaseIndex(3)
    index.add([r for r, _ in rows], np.array([v for _, v in rows], dtype=np.float32))
    return index


QUERY = np.array([1.0, 0.0, 0.0])
ids = lambda hits: [h.record["case_id"] for h in hits]


class CaseLabelTests(unittest.TestCase):
    def test_only_listed_choices_are_accepted(self):
        CaseLabels(disease_site="Lung & Thoracic", prior_rt="Re-irradiation (overlap)")
        self.assertEqual(CaseLabels().setting_group, NOT_SPECIFIED)
        with self.assertRaises(ValueError):
            CaseLabels(disease_site="Lung")

    def test_prior_rt_groups(self):
        self.assertEqual(prior_rt_group("None"), "No prior RT")
        self.assertEqual(prior_rt_group("Re-irradiation (overlap)"), "Prior RT")
        self.assertEqual(prior_rt_group(NOT_SPECIFIED), NOT_SPECIFIED)


class IndexTests(unittest.TestCase):
    def test_search_filters_excludes_and_orders_by_similarity(self):
        index = build([(record("A", "P1"), [1, 0, 0]), (record("B", "P2"), [0.9, 0.1, 0]),
                       (record("C", "P3"), [0, 1, 0]), (record("D", "P4", site="CNS"), [1, 0, 0])])
        self.assertEqual(ids(index.search(QUERY, {"disease_site": "Breast"}, {}, 2)), ["A", "B"])
        self.assertEqual(ids(index.search(QUERY, {"disease_site": "Breast"}, {"patient_id": {"P1"}}, 5)), ["B", "C"])
        self.assertEqual(index.search(QUERY, {"disease_site": "Skin"}, {}, 5), [])

    def test_add_replaces_by_case_id_and_changes_the_version(self):
        index = build([(record("A", "P1"), [1, 0, 0])])
        before = index.version()
        index.add([record("A", "P1", setting="Salvage"), record("B", "P2")], np.array([[0, 1, 0], [0, 0, 1]]))
        self.assertEqual(len(index.records), 2)
        self.assertEqual(index.records[0]["setting_group"], "Salvage")
        self.assertNotEqual(index.version(), before)

    def test_save_and_load_keep_rows_aligned(self):
        index = build([(record("A", "P1"), [1, 0, 0]), (record("B", "P2"), [0, 1, 0])])
        with tempfile.TemporaryDirectory() as folder:
            index.save(folder)
            loaded = NumpyPastCaseIndex.load(folder)
        self.assertEqual(loaded.version(), index.version())
        self.assertEqual(ids(loaded.search(QUERY, {}, {}, 1)), ["A"])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.index = build([
            (record("SELF", "P0"), [1, 0, 0]),
            (record("TWIN", "P0"), [1, 0, 0]),                          # same patient, other encounter
            (record("ADJ", "P1"), [0.6, 0.4, 0]),
            (record("SALV", "P2", setting="Salvage"), [0.9, 0.1, 0]),
            (record("BOOST", "P3", setting="Salvage", encounter="Boost"), [0.5, 0.5, 0]),
            (record("CNS", "P4", site="CNS"), [1, 0, 0]),
        ])

    def select(self, **labels):
        defaults = {"disease_site": "Breast", "setting_group": "Adjuvant", "encounter_type": "New start"}
        return ids(select_examples(self.index, QUERY, CaseLabels(**{**defaults, **labels}), "SELF", "P0"))

    def test_same_setting_first_then_same_site_never_another_site(self):
        # Only ADJ shares the setting; SALV fills the second slot from the same site; CNS is never chosen
        self.assertEqual(self.select(), ["ADJ", "SALV"])

    def test_self_and_same_patient_are_never_chosen(self):
        self.assertNotIn("TWIN", self.select(setting_group=NOT_SPECIFIED))
        self.assertNotIn("SELF", self.select(setting_group=NOT_SPECIFIED))

    def test_same_encounter_type_ranks_first(self):
        self.assertEqual(self.select(setting_group="Salvage", encounter_type="Boost"), ["BOOST", "SALV"])

    def test_no_case_is_chosen_twice(self):
        # ADJ is the closest Breast case overall, so the site-only tier would return it again
        index = build([(record("ADJ", "P1"), [1, 0, 0]), (record("SALV", "P2", setting="Salvage"), [0.5, 0.5, 0])])
        labels = CaseLabels(disease_site="Breast", setting_group="Adjuvant")
        self.assertEqual(ids(select_examples(index, QUERY, labels)), ["ADJ", "SALV"])

    def test_same_prior_rt_group_ranks_before_closer_cases(self):
        index = build([(record("CLOSE", "P1"), [1, 0, 0]),
                       (record("REIRR", "P2", prior="Prior RT"), [0.2, 0.8, 0]),
                       (record("OTHER", "P3", setting="Salvage", prior="Prior RT"), [1, 0, 0])])
        labels = CaseLabels(disease_site="Breast", setting_group="Adjuvant", prior_rt="Re-irradiation (overlap)")
        # Same setting and prior RT first, then same setting; a closer case from another setting waits
        self.assertEqual(ids(select_examples(index, QUERY, labels)), ["REIRR", "CLOSE"])
        unspecified = CaseLabels(disease_site="Breast", setting_group="Adjuvant")
        self.assertEqual(ids(select_examples(index, QUERY, unspecified)), ["CLOSE", "REIRR"])

    def test_similarity_floor_applies_in_every_tier(self):
        index = build([(record("NEAR", "P1"), [1, 0, 0]), (record("FAR", "P2"), [0, 1, 0]),
                       (record("FAR_SALV", "P3", setting="Salvage"), [0.1, 1, 0])])
        labels = CaseLabels(disease_site="Breast", setting_group="Adjuvant")
        self.assertEqual(ids(select_examples(index, QUERY, labels)), ["NEAR", "FAR"])
        # The site-only tier must not bring back a weak case the floor removed
        self.assertEqual(ids(select_examples(index, QUERY, labels, min_similarity=0.5)), ["NEAR"])

    def test_unspecified_site_gives_no_examples(self):
        self.assertEqual(self.select(disease_site=NOT_SPECIFIED), [])


if __name__ == "__main__":
    unittest.main()
