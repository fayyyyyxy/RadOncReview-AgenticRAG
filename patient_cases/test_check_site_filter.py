"""Site filter check on a synthetic answered case; no AWS."""

import unittest

from check_site_filter import check_case


def result(passages, retrieved=(), site_filter=True):
    return {
        "case_id": "D1", "status": "response_generated",
        "case_labels": {"disease_site": "Lung & Thoracic", "setting_group": "Definitive",
                        "prior_rt": "None", "encounter_type": "New start"},
        "response": {"context_passages": passages, "routing": {"site_filter": site_filter},
                     "task_plan": {"sub_tasks": [{"retrieved_chunks": list(retrieved)}]}},
    }


NOTE = {"source_type": "review_notes", "category": "Thorax"}
PAPER = {"source_type": "clinical_trial", "category": "Lung"}


class CheckSiteFilterTests(unittest.TestCase):
    def test_evidence_inside_the_site_passes(self):
        row = check_case(result([NOTE, PAPER], [NOTE]))
        self.assertEqual((row["outside_scope"], row["review_notes"], row["trial_papers"]), (0, 1, 1))

    def test_other_site_evidence_is_counted_even_if_never_shown(self):
        other = {"source_type": "clinical_trial", "category": "Head & Neck"}
        self.assertEqual(check_case(result([NOTE], [other]))["outside_scope"], 1)
        self.assertFalse(check_case(result([NOTE], site_filter=False))["site_filter_on"])


if __name__ == "__main__":
    unittest.main()
