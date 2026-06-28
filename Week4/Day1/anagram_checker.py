import os
from typing import List, Optional


class AnagramChecker:
    """Minimal loader for a wordlist file.

    It simply loads `sowpods.txt` into
    `self.words` (a list of normalized words) so other parts of the program can
    search it later.
    """

    def __init__(self, wordlist_path: Optional[str] = None, encoding: str = "utf-8"):
        if wordlist_path is None:
            base = os.path.dirname(__file__)
            wordlist_path = os.path.join(base, "sowpods.txt")

        self.wordlist_path = wordlist_path

        try:
            with open(self.wordlist_path, "r", encoding=encoding) as fh:
                # here we normalize to lowercase
                self.words: List[str] = [line.strip() for line in fh if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Wordlist not found: {self.wordlist_path}")

    def is_valid_word(self, word: str) -> bool:
        """Return True if `word` exists in the loaded wordlist (case-insensitive)."""
        # ensure a lowercase set exists for fast checks
        if not hasattr(self, "words_set"):
            self.words_set = set(w.lower() for w in self.words)
        if not word:
            return False
        return word.lower() in self.words_set

    def is_anagram(self, word1: str, word2: str) -> bool:
        """Return True if `word1` and `word2` contain the same letters but are not identical."""
        if not word1 or not word2:
            return False
        w1 = word1.lower()
        w2 = word2.lower()
        if w1 == w2:
            return False
        return sorted(w1) == sorted(w2)

    def get_anagrams(self, word: str) -> List[str]:
        """Return a list of anagrams for `word` found in the loaded wordlist.

        Returns lowercased words and excludes the original word.
        """
        if not word:
            return []
        target = word.lower()
        key = sorted(target)
        results: List[str] = []
        for w in self.words:
            lw = w.lower()
            if lw == target:
                continue
            if sorted(lw) == key:
                results.append(lw)
        return results
