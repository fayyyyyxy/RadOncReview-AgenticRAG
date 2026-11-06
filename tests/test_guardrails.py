"""Which questions the input guardrail blocks for a blocked term."""

import unittest

from config import GuardrailConfig
from core.guardrails import GuardrailViolation, InputGuardrail


def blocked(text):
    return GuardrailViolation.BLOCKED_CONTENT in InputGuardrail(GuardrailConfig()).check(text).violations


class BlockedTermTests(unittest.TestCase):
    def test_a_word_containing_a_blocked_term_passes(self):
        # "DOB" sits inside "endobronchial" and "dobutamine", "MRN" inside "mRNA"
        for text in ("Palliative RT for endobronchial obstruction from NSCLC",
                     "Dobutamine stress echo before chemoradiation",
                     "HER2 mRNA expression in breast cancer"):
            with self.subTest(text=text):
                self.assertFalse(blocked(text))

    def test_a_blocked_term_on_its_own_is_still_caught(self):
        for text in ("DOB: 01/02/1950", "dob 01/02/1950", "MRN 12345678", "Her date of birth is in the chart"):
            with self.subTest(text=text):
                self.assertTrue(blocked(text))


if __name__ == "__main__":
    unittest.main()
