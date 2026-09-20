"""Synthesizer: idea -> draft with real claims, no invented numbers."""
import unittest

from jevgraphic.synth import synthesize
from jevgraphic.draft import numbers_in

EVIDENCE = ("open-Jev reaches 65/70 = 92.9% on the gold set. "
            "The classifier arm scores 66/70 = 94.3%. "
            "Warm inference is 6 ms per posting offline. "
            "Judgment as a service now has a local price of zero.")


class TestSynth(unittest.TestCase):
    def test_title_subtitle_split(self):
        d = synthesize("open-Jev: local judgment at $0", EVIDENCE)
        self.assertEqual(d.title, "open-Jev")
        self.assertIn("local judgment", d.subtitle)

    def test_claims_come_from_evidence(self):
        d = synthesize("judgment locally for nothing", EVIDENCE)
        self.assertTrue(d.claims)
        # synthesizer may only emit text that exists in evidence (modulo markup)
        norm = EVIDENCE.replace("**", "")
        for c in d.claims:
            self.assertIn(c.text[:30], norm)

    def test_no_evidence_still_drafts(self):
        d = synthesize("a thing: with a subtitle", "")
        self.assertEqual(d.title, "a thing")
        self.assertTrue(d.subtitle)

    def test_numbered_claims_ranked_first(self):
        d = synthesize("accuracy judgment", EVIDENCE)
        self.assertTrue(numbers_in(d.claims[0].text))


if __name__ == "__main__":
    unittest.main()
