import os
from anagram_checker import AnagramChecker
from typing import List


def prompt_menu() -> str:
    print("\nAnagram Finder")
    print("1) Input a word")
    print("2) Exit")
    return input("Choose an option (1-2): ").strip()


def validate_input(raw: str) -> (bool, str):
    """Return (is_valid, cleaned_word_or_error_message)."""
    if raw is None:
        return False, "No input provided"
    text = raw.strip()
    if not text:
        return False, "Empty input"

    tokens: List[str] = text.split()
    if len(tokens) != 1:
        return False, "Please enter exactly one word (no spaces)"

    word = tokens[0]
    if not word.isalpha():
        return False, "Only alphabetic characters are allowed"

    return True, word


def format_anagram_output(word: str, is_valid: bool, anagrams: List[str]) -> str:
    header = f'YOUR WORD :"{word.upper()}"\n'
    validity = (
        "this is a valid English word."
        if is_valid
        else "this is NOT a valid English word."
    )
    if anagrams:
        anag_text = ", ".join(anagrams)
    else:
        anag_text = "(no anagrams found)"
    body = f"{validity}\nAnagrams for your word: {anag_text}"
    return header + body


def main():
    # ensure we pass the correct sowpods path (same directory as this script)
    base = os.path.dirname(__file__)
    wordlist = os.path.join(base, "sowpods.txt")
    try:
        checker = AnagramChecker(wordlist_path=wordlist)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    while True:
        choice = prompt_menu()
        if choice == "2":
            print("Goodbye.")
            break
        if choice != "1":
            print("Invalid choice, please enter 1 or 2.")
            continue

        raw = input("Enter a word: ")
        valid, payload = validate_input(raw)
        if not valid:
            print("Error:", payload)
            continue

        word = payload
        is_valid_word = checker.is_valid_word(word)
        anagrams = checker.get_anagrams(word)
        print(format_anagram_output(word, is_valid_word, anagrams))


if __name__ == "__main__":
    main()
