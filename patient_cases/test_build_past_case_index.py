"""Past-case index records and incremental updates on synthetic cases; no AWS."""

import unittest

import numpy as np

from build_past_case_index import relabelled, stale, to_record
from core.past_cases import KEYS, NumpyPastCaseIndex
from load_scenarios import development_examples
from test_load_scenarios import labelled


def example(case_id="D1", **changes):
    row = {**labelled(case_id, "P1", "Breast"), "Setting": "Adjuvant", **changes}
    return development_examples([row])[0]


class BuildIndexTests(unittest.TestCase):
    def test_record_carries_routing_keys_and_payload(self):
        record = to_record(example())
        self.assertTrue(set(KEYS) <= set(record))
        self.assertEqual((record["setting_group"], record["patient_id"]), ("Adjuvant", "P1"))
        self.assertIn("attending_plan", record)

    def test_ineligible_cases_are_refused(self):
        case = example()
        case["governance"]["split"] = "Locked test"
        with self.assertRaisesRegex(ValueError, "not an eligible development case"):
            to_record(case)

    def test_only_new_or_changed_cases_are_embedded_again(self):
        first, second = to_record(example("D1")), to_record(example("D2"))
        index = NumpyPastCaseIndex(3)
        index.add([first, second], np.eye(3)[:2])
        changed = to_record({**example("D2"), "search_text": "Edited vignette\n\nSynthetic question"})
        new = to_record(example("D3"))
        self.assertEqual([r["case_id"] for r in stale([first, changed, new], index)], ["D2", "D3"])

    def test_label_fix_updates_the_record_without_embedding(self):
        first = to_record(example("D1"))
        index = NumpyPastCaseIndex(3)
        index.add([first], np.array([[0, 1, 0]]))
        fixed = to_record(example("D1", **{"Prior RT": "Re-irradiation (overlap)"}))
        self.assertEqual(stale([fixed], index), [])
        changed, vectors = relabelled([fixed], index)
        self.assertEqual([r["prior_rt_group"] for r in changed], ["Prior RT"])
        np.testing.assert_allclose(vectors, [[0, 1, 0]])


if __name__ == "__main__":
    unittest.main()
