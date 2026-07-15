# 🌟 Exercise 1: Random Sentence Generator (~20 minutes)
# Build a program that reads words from a file and generates random sentences.

# Functions to implement:

# Function	Details
# get_words_from_file(file_path)	Opens the file, reads content, splits into words, returns the list
# get_random_sentence(length)	Takes a number, picks that many random words, joins with spaces, returns lowercase string
# main()	Asks for sentence length (2-20), validates input with try/except, calls the other functions
# Rules:

# main() should ask the user for a number between 2 and 20 (inclusive)
# If the user types something that isn't a number → catch ValueError, print error message
# If the number is outside 2-20 → print an error message (not an exception — just a check)
# Use random.choice() to pick random words
# The sentence should be all lowercase
# Example output:

# Enter sentence length (2-20): 5
# Generated sentence: gentle river apple swift code

# Enter sentence length (2-20): hello
# Invalid input! Please enter a number.

# Enter sentence length (2-20): 25
# Please enter a number between 2 and 20.


# 🌟 Exercise 1 — Random Sentence Generator
# 🌟 Exercise 1 — Random Sentence Generator
import random

FILE_PATH = "Week3\\Day4\\words.txt"


def get_words_from_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        words = content.split()
        return words


def get_random_sentence(length):
    words = get_words_from_file(FILE_PATH)

    random_words = []

    for i in range(length):
        random_words.append(random.choice(words))

    sentence = " ".join(random_words)

    return sentence.lower()


def main():
    try:
        length = int(input("Enter sentence length (2-20): "))

        if length < 2 or length > 20:
            print("Please enter a number between 2 and 20.")
        else:
            sentence = get_random_sentence(length)
            print(f"Generated sentence: {sentence}")

    except ValueError:
        print("Invalid input! Please enter a number.")


main()
