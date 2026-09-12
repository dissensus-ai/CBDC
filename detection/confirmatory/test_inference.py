"""Gatekeeping regression tests for the September 2026 defect."""

import unittest

from inference import summarize_hypotheses


class Gatekeeping(unittest.TestCase):
    def test_unset_tolerance_never_opens_h2(self):
        h1, h2 = summarize_hypotheses([7, 8, 9], [20, 21, 22], None)
        self.assertEqual(h1["verdict"], "ESTIMATE_ONLY")
        self.assertEqual(h2["h1_verdict"], "ESTIMATE_ONLY")
        self.assertFalse(h2["confirmatory"])
        self.assertEqual(h2["status"], "DESCRIPTIVE_ONLY")

    def test_established_noninferiority_can_open_h2(self):
        h1, h2 = summarize_hypotheses([1, 2, 3], [20, 21, 22], 10)
        self.assertEqual(h1["verdict"], "NON_INFERIOR")
        self.assertTrue(h2["confirmatory"])

    def test_explicitly_descriptive_lock_stays_descriptive(self):
        h1, h2 = summarize_hypotheses(
            [1, 2, 3], [20, 21, 22], 10, h2_confirmatory=False
        )
        self.assertEqual(h1["verdict"], "NON_INFERIOR")
        self.assertFalse(h2["confirmatory"])

    def test_failed_h1_does_not_open_h2(self):
        _, h2 = summarize_hypotheses([20, 21, 22], [20, 21, 22], 1)
        self.assertFalse(h2["confirmatory"])


if __name__ == "__main__":
    unittest.main()
