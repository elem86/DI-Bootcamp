import random

# Exercise 1: Concatenate lists
# Instructions
# Write code that concatenates two lists together without using the + sign.

a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]

c = a.copy()  # Create a copy of list a
c.extend(b)  # Extend the copy with the elements of list b
print(c)

# Exercise 2: Range of numbers
# Instructions
# Create a loop that goes from 1500 to 2500 and prints all multiples of 5 and 7.

for num in range(1500, 2501):
    if num % 5 == 0 and num % 7 == 0:
        print(num)


# Exercise 3: Check the index
# Instructions
# Using this variable

# names = ['Samus', 'Cortana', 'V', 'Link', 'Mario', 'Cortana', 'Samus']
# Ask a user for their name, if their name is in the names list print out the index of the first occurence of the name.

# Example: if input is 'Cortana' we should be printing the index 1

names = ["Samus", "Cortana", "V", "Link", "Mario", "Cortana", "Samus"]
user_name = input("Please enter your name: ")
if user_name in names:
    index = names.index(user_name)
    print(f"{user_name} is found at index {index}.")
else:
    print("Name not found in the list.")


# Exercise 4: Greatest Number
# Instructions
# Ask the user for 3 numbers and print the greatest number.

# Test Data
# Input the 1st number: 25
# Input the 2nd number: 78
# Input the 3rd number: 87

# The greatest number is: 87

number = int(input("Enter a number: "))

greatest = number  # Initialize greatest with the first number

for i in range(2):  # We already have the first number, so we need to ask for 2 more
    number = int(input("Enter another number: "))
    if number > greatest:
        greatest = number

print(f"The greatest number is: {greatest}")

# Exercise 5: The Alphabet
# Instructions
# Create a string of all the letters in the alphabet
# Loop over each letter and print a message that contains the letter and whether its a vowel or a consonant.

alphabet = "abcdefghijklmnopqrstuvwxyz"

while True:
    user_input = input("Enter a letter: ").lower()
    if user_input in "aeiou":
        print(f"{user_input} is a vowel.")
        break
    elif user_input in alphabet:
        print(f"{user_input} is a consonant.")
        break
    else:
        print("Please enter a valid letter.")

# Exercise 6: Words and letters
# Instructions
# Ask a user for 7 words, store them in a list named words.
# Ask the user for a single character, store it in a variable called letter.
# Loop through the words list and print the index of the first appearence of the letter variable in each word of the list.
# If the letter doesn’t exist in one of the words, print a friendly message with the word and the letter.

words = []
for i in range(7):
    word = input("Enter a word: ")
    words.append(word)
letter = input("Enter a single character: ")
for word in words:
    if letter in word:
        index = word.index(letter)
        print(f"The letter '{letter}' is found in the word '{word}' at index {index}.")
    else:
        print(f"The letter '{letter}' is not found in the word '{word}'.")


# Exercise 7: Min, Max, Sum
# Instructions
# Create a list of numbers from one to one million and then use min() and max() to make sure your list actually starts at one and ends at one million. Use the sum() function to see how quickly Python can add a million numbers.

numbers = list(range(1, 1000001))
print(f"Minimum number: {min(numbers)}")
print(f"Maximum number: {max(numbers)}")
print(f"Sum of all numbers: {sum(numbers)}")


# Exercise 8 : List and Tuple
# Instructions
# Write a program which accepts a sequence of comma-separated numbers. Generate a list and a tuple which contain every number.

# Suppose the following input is supplied to the program: 34,67,55,33,12,98

# Then, the output should be:

# ['34', '67', '55', '33', '12', '98']
# ('34', '67', '55', '33', '12', '98')


list_input = input("Enter a sequence of comma-separated numbers: ")
number_list = list_input.split(",")  # Split the input string into a list
number_tuple = tuple(number_list)  # Convert the list to a tuple
print(f"List: {number_list}")
print(f"Tuple: {number_tuple}")


# Exercise 9 : Random number
# Instructions
# Ask the user to input a number from 1 to 9 (including).
# Get a random number between 1 and 9. Hint: random module.
# If the user guesses the correct number print a message that says Winner.
# If the user guesses the wrong number print a message that says better luck next time.
# Bonus: use a loop that allows the user to keep guessing until they want to quit.
# Bonus 2: on exiting the loop tally up and display total games won and lost.

wins = 0
losses = 0

while True:
    user_guess = int(input("Guess a number between 1 and 9: "))

    random_number = random.randint(1, 9)

    if user_guess == random_number:
        print("Winner")
        wins += 1
    else:
        print("Better luck next time")
        losses += 1

    play_again = input("Do you want to keep playing? (yes/no): ")

    if play_again.lower() == "no" or play_again.lower() == "n":
        break

print("Total games won:", wins)
print("Total games lost:", losses)
