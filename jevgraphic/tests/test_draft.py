import unittest

from jevgraphic.draft import parse_draft


class TestDraft(unittest.TestCase):
    def test_parse(self):
        d = parse_draft("# T\na subtitle\n- one\n- 50% two\nprose line\n## Section\n- three")
        self.assertEqual(d.title, "T")
        self.assertEqual(d.subtitle, "a subtitle")
        self.assertEqual([c.text for c in d.claims], ["one", "50% two", "prose line", "three"])

    def test_bold_unwrapped(self):
        d = parse_draft("# T\n- **92.9%** accuracy")
        self.assertEqual(d.claims[0].text, "92.9% accuracy")


if __name__ == "__main__":
    unittest.main()
