import random


# This is the list of possible words or phrases for the game.
wordslist = [
    "correction",
    "childish",
    "beach",
    "python",
    "assertive",
    "interference",
    "complete",
    "share",
    "credit card",
    "rush",
    "south",
]

# The computer chooses one random word from the list.
word = random.choice(wordslist)

### YOUR CODE STARTS FROM HERE ###

# Each wrong guess adds one body part.
body_parts = ["head", "body", "left arm", "right arm", "left leg", "right leg"]

# This list stores all the letters the player already guessed.
guessed_letters = []

# This counts how many wrong guesses the player made.
wrong_guesses = 0


def get_hidden_word():
    # This function builds the word that the player sees.
    # Unknown letters become "*", guessed letters are shown, and spaces stay spaces.
    hidden_word = ""

    for letter in word:
        if letter == " ":
            hidden_word += " "
        elif letter in guessed_letters:
            hidden_word += letter
        else:
            hidden_word += "*"

    return hidden_word


# Show the player the starting hidden word.
print("Welcome to Hangman!")
print(get_hidden_word())

# The game continues while:
# 1. the player still has body parts left
# 2. the word still has hidden letters
while wrong_guesses < len(body_parts) and "*" in get_hidden_word():
    # Ask the player for a letter and make it lowercase.
    guess = input("Guess a letter: ").lower()

    # The player must type exactly one alphabet letter.
    if len(guess) != 1 or not guess.isalpha():
        print("Please enter one letter only.")
        continue

    # The player is not allowed to guess the same letter twice.
    if guess in guessed_letters:
        print("You already guessed that letter. Try another one.")
        continue

    # Add the valid guess to the list of guessed letters.
    guessed_letters.append(guess)

    # If the guessed letter is inside the word, it will be revealed.
    if guess in word:
        print("Good guess!")
    else:
        # If the guessed letter is not inside the word, add the next body part.
        print(f"Wrong guess! The computer adds the {body_parts[wrong_guesses]}.")
        wrong_guesses += 1

    # Show the current game state after each guess.
    print(get_hidden_word())
    print(f"Guessed letters: {', '.join(guessed_letters)}")
    print(f"Wrong guesses: {wrong_guesses}/{len(body_parts)}")

# When the loop ends, check if the player revealed the whole word.
if "*" not in get_hidden_word():
    print(f"You won! The word was: {word}")
else:
    print(f"You lost! The word was: {word}")
