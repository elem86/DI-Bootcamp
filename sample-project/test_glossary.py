"""Checks for the glossary tool. Run them with:

    python test_glossary.py

Right now one check fails on purpose — that's the change you'll make with
Claude Code in the "Getting Started with Claude Code" tutorial.
"""

import unittest

from glossary import look_up


class TestGlossary(unittest.TestCase):
    def test_exact_term_is_found(self):
        # A term typed exactly as stored returns its definition.
        self.assertIn("text unit", look_up("token"))

    def test_two_word_term_is_found(self):
        self.assertIn("tokens", look_up("context window"))

    def test_unknown_term_is_friendly(self):
        # Unknown terms should suggest what's available, not crash.
        self.assertIn("No entry", look_up("banana"))

    def test_lookup_is_case_insensitive(self):
        # A student typing "Token" or "TOKEN" should still get the definition.
        # This one FAILS until you make look_up() case-insensitive.
        self.assertIn("text unit", look_up("Token"))
        self.assertIn("text unit", look_up("TOKEN"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
