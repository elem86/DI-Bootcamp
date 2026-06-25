import re
import string


class Text:
    def __init__(self, text):
        self.text = text

    def word_frequency(self, word):
        words = self.text.split()
        count = words.count(word)
        return count if count > 0 else None

    def most_common_word(self):
        words = self.text.split()
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1

        if not freq:
            return None

        most_common = max(freq.items(), key=lambda item: item[1])[0]
        return most_common

    def unique_words(self):
        words = self.text.split()
        return list(set(words))

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(text)


class TextModification(Text):
    def remove_punctuation(self):
        translator = str.maketrans("", "", string.punctuation)
        return self.text.translate(translator)

    def remove_stop_words(self):
        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "while",
            "with",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "from",
            "by",
            "about",
            "as",
            "into",
            "over",
            "after",
            "before",
            "between",
            "than",
            "that",
            "which",
            "this",
            "these",
            "those",
            "it",
            "its",
            "he",
            "she",
            "they",
            "them",
            "his",
            "her",
            "their",
            "you",
            "your",
            "we",
            "us",
            "our",
            "my",
            "me",
            "i",
        }
        words = self.text.split()
        filtered = [word for word in words if word.lower() not in stop_words]
        return " ".join(filtered)

    def remove_special_characters(self):
        return re.sub(r"[^A-Za-z0-9\s]", "", self.text)


if __name__ == "__main__":
    sample = "Hello, world! This is a sample text -- with punctuation, stop words, and special_chars."
    mod = TextModification(sample)

    print("Original:", mod.text)
    print("No punctuation:", mod.remove_punctuation())
    print("No stop words:", mod.remove_stop_words())
    print("No special characters:", mod.remove_special_characters())
