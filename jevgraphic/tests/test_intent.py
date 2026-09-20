"""Intent routing is deterministic and scores real options."""
import unittest

from jevgraphic.art import art_names
from jevgraphic.intent import LocalIntent
from jevgraphic.layouts import layout_names
from jevgraphic.model import Draft, Claim
from jevgraphic.styles import style_names


class TestIntent(unittest.TestCase):
    def _draft(self):
        return Draft(title="t", claims=[Claim(text="a claim")])

    def test_deterministic(self):
        a = LocalIntent().decide("benchmark report", self._draft())
        b = LocalIntent().decide("benchmark report", self._draft())
        self.assertEqual(a.layout, b.layout)
        self.assertEqual(a.style, b.style)
        self.assertEqual(a.art, b.art)

    def test_options_are_real(self):
        d = LocalIntent().decide("a comparison of two models", self._draft())
        self.assertIn(d.layout, layout_names())
        self.assertIn(d.style, style_names())
        self.assertIn(d.art, art_names())

    def test_compare_idea_picks_compare_layout(self):
        d = LocalIntent().decide(
            "versus two columns comparison before after", self._draft())
        self.assertEqual(d.layout, "compare")

    def test_all_layouts_reachable(self):
        # different ideas should exercise different layouts
        picks = set()
        for idea in ("launch announcement hero number",
                     "metrics kpi dashboard tiles report",
                     "step by step process how it works ordered",
                     "versus two columns comparison before after",
                     "timeline stream of events history feed"):
            picks.add(LocalIntent().decide(idea, self._draft()).layout)
        self.assertGreaterEqual(len(picks), 3)


if __name__ == "__main__":
    unittest.main()
