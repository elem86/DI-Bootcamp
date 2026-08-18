"""
Week 1 - Day 1 exercises
(Test file created to verify the Student Progress Coach pipeline end-to-end.)
"""


def word_lengths(words):
    """Exercise 1: return a dict mapping each word to its length."""
    return {word: len(word) for word in words}


def safe_divide(a, b):
    """Exercise 2: divide two numbers, handling division by zero."""
    try:
        return a / b
    except ZeroDivisionError:
        return None


if __name__ == "__main__":
    print(word_lengths(["python", "loop", "dict"]))
    print(safe_divide(10, 2))
    print(safe_divide(10, 0))
