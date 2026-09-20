"""The gate's contract: no number reaches the canvas unless it exists
verbatim in the evidence document."""
import unittest

from jevgraphic.draft import numbers_in, parse_draft
from jevgraphic.gate import LocalGate
from jevgraphic.model import Claim

EVIDENCE = """open-Jev reaches 65/70 = 92.9% on the gold set.
The classifier arm scores 66/70 = 94.3% and agrees on 99.39% of 2,631 postings.
Warm inference is 6 ms per posting. The viral artifact scored 77.1%.
"""


class TestGate(unittest.TestCase):
    def test_verbatim_number_kept(self):
        d = parse_draft("# t\n- reaches 92.9% on the gold set")
        d, r = LocalGate().decide(d, EVIDENCE)
        self.assertEqual(r.kept, 1)
        self.assertEqual(r.refused, 0)
        self.assertEqual(r.cost_usd, 0.0)

    def test_unverified_number_refused(self):
        d = parse_draft("# t\n- reaches 99.9% on the gold set")
        d, r = LocalGate().decide(d, EVIDENCE)
        self.assertEqual(r.kept, 0)
        self.assertEqual(r.refused, 1)
        self.assertIn("99.9%", d.claims[0].note)

    def test_comma_form_matches(self):
        d = parse_draft("# t\n- agrees on 99.39% of 2631 postings")
        d, r = LocalGate().decide(d, EVIDENCE)
        self.assertEqual(r.kept, 1)

    def test_no_evidence_refuses_numbers(self):
        d = parse_draft("# t\n- 50% of anything")
        d, r = LocalGate().decide(d, "")
        self.assertEqual(r.kept, 0)
        self.assertEqual(d.claims[0].note, "no evidence supplied")

    def test_no_evidence_keeps_prose(self):
        d = parse_draft("# t\n- local judgment costs nothing")
        d, r = LocalGate().decide(d, "")
        self.assertEqual(r.kept, 1)

    def test_copied_count(self):
        d = parse_draft("# t\n- Warm inference is 6 ms per posting.")
        d, r = LocalGate().decide(d, EVIDENCE)
        self.assertEqual(r.copied, 1)

    def test_numbers_in(self):
        self.assertIn("92.9%", numbers_in("hit 92.9% accuracy"))
        self.assertIn("$0", numbers_in("for $0"))
        self.assertIn("2,631", numbers_in("2,631 postings"))

    def test_tokenizer_never_gloms_following_word(self):
        # regression: "54/54, service" used to yield the token "54, s"
        toks = numbers_in("staff role 12/14, generic job 54/54, service lead 0/2")
        self.assertNotIn("54, s", toks)
        self.assertIn("54", toks)

    def test_comma_list_claim_verifies(self):
        ev = EVIDENCE + "Per-class: staff role 12/14, generic job 54/54, service lead 0/2."
        d = parse_draft("# t\n- Per-class: staff role 12/14, generic job 54/54, service lead 0/2.")
        d, r = LocalGate().decide(d, ev)
        self.assertEqual(r.kept, 1)


if __name__ == "__main__":
    unittest.main()
