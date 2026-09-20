"""Composition invariants: valid SVG, refused claims never drawn, every
rendered number verified."""
import re
import unittest

from jevgraphic.compose import compose_svg
from jevgraphic.draft import numbers_in, parse_draft
from jevgraphic.gate import LocalGate, _evidence_numbers
from jevgraphic.intent import LocalIntent
from jevgraphic.layouts import layout_names
from jevgraphic.model import Decision, GateReceipt
from jevgraphic.styles import style_names

EVIDENCE = ("open-Jev reaches 65/70 = 92.9% on the gold set. "
            "The classifier arm scores 66/70 = 94.3% and agrees on 99.39% of 2,631 postings. "
            "Warm inference is 6 ms per posting. The viral artifact scored 77.1%. ")

DRAFT = """# open-Jev: Local Judgment at $0
reproducing a hosted decision model locally, for nothing
- 92.9% accuracy for the 0.5B LM arm trained overnight on CPU
- 94.3% for the 3 MB classifier, 99.39% agreement on the live stream
- 6 ms warm inference per posting, numpy only
- 77.1% for the viral artifact — the majority-class prior
- 88.8% this number is invented and must never render
"""


class TestCompose(unittest.TestCase):
    def _build(self, layout="poster", style="blueprint-technical"):
        draft = parse_draft(DRAFT)
        draft, receipt = LocalGate().decide(draft, EVIDENCE)
        dec = Decision(layout=layout, style=style, art="sunburst", backend="test")
        return compose_svg(draft, dec, receipt), draft

    def test_valid_svg(self):
        svg, _ = self._build()
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        import xml.dom.minidom as md
        md.parseString(svg)  # raises if malformed

    def test_invented_number_never_renders(self):
        for layout in layout_names():
            for style in style_names():
                svg, draft = self._build(layout, style)
                self.assertNotIn("88.8%", svg, f"{layout}/{style} leaked a refused number")

    def test_rendered_numbers_subset_of_evidence(self):
        ev = _evidence_numbers(EVIDENCE) | _evidence_numbers(DRAFT.split("\n")[0])
        extra_ok = {str(i) for i in range(10)}  # layout furniture: step ordinals, glyphs
        for layout in layout_names():
            svg, draft = self._build(layout)
            # check visible text nodes only, not markup attributes
            for txt in re.findall(r">([^<>]+)<", svg):
                if "numbers verified" in txt or "chosen by jev" in txt:
                    continue  # gate provenance line — describes the gate, not the subject
                for n in numbers_in(txt):
                    if n in extra_ok:
                        continue
                    ok = n in ev or n.replace(",", "") in ev
                    self.assertTrue(ok, f"{layout}: unverified number {n!r} in {txt!r}")

    def test_refused_claim_not_drawn(self):
        svg, draft = self._build()
        self.assertFalse(draft.claims[-1].kept)
        self.assertNotIn("invented", svg)


if __name__ == "__main__":
    unittest.main()
