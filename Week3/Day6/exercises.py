# 🌟 Exercise 1: Random Sentence Generator
# Goal: Create a program that generates a random sentence of a specified length from a word list.


# Key Python Topics:

# File handling (open(), read())
# Lists
# Random number generation (random.choice())
# String manipulation (split(), join(), lower())
# Error handling (try, except)
# Input validation

import random


# Instructions:

# Download the provided word list and save it in your development directory.
# Create a function to read the words from the file.
# Create a function to generate a random sentence of a given length.
# Create a main function to handle user input and program flow.


# Step 1: Create the get_words_from_file function

# Create a function named get_words_from_file that takes the file path as an argument.
# Open the file in read mode ("r").
# Read the file content.
# Split the content into a list of words.
# Return the list of words.


# Step 2: Create the get_random_sentence function

# Create a function named get_random_sentence that takes the sentence length as an argument.
# Call get_words_from_file to get the list of words.
# Select a random word from the list length times.
# Create a sentence with the selected words.
# Convert the sentence to lowercase.
# Return the sentence.


# Step 3: Create the main function

# Create a function named main.
# Print a message explaining the program’s purpose.
# Ask the user for the desired sentence length.
# Validate the user input:
# Check if it is an integer.
# Check if it is between 2 and 20 (inclusive).
# If the input is invalid, print an error message and exit.
# If the input is valid, call get_random_sentence with the length and print the generated sentence.


def get_words_from_file(filename="Week3/Day6/words.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_random_sentence(length):
    words = get_words_from_file()
    selected_words = [random.choice(words) for _ in range(length)]
    sentence = " ".join(selected_words).lower()
    return sentence


def main():
    print("Random sentence generator: create a sentence from a word list.")
    user_input = input("Enter desired sentence length (2-20): ")

    try:
        length = int(user_input)
    except ValueError:
        print("Invalid input: please enter an integer.")
        return

    if length < 2 or length > 20:
        print("Invalid input: length must be between 2 and 20.")
        return

    sentence = get_random_sentence(length)
    print(sentence)


if __name__ == "__main__":
    main()


# 🌟 Exercise 2: Working with JSON
# Goal: Access a nested key in a JSON string, add a new key, and save the modified JSON to a file.


# Key Python Topics:

# JSON parsing (json.loads())
# JSON serialization (json.dump())
# Dictionaries
# File handling (open())


# Instructions:

# Using the follow code:

# import json
# sampleJson = """{
#    "company":{
#       "employee":{
#          "name":"emma",
#          "payable":{
#             "salary":7000,
#             "bonus":800
#          }
#       }
#    }
# }"


# Access the nested “salary” key.
# Add a new key “birth_date” wich value is of format “YYYY-MM-DD”, to the “employee” dictionary: "birth_date": "YYYY-MM-DD".
# Save the modified JSON to a file.


# Step 1: Load the JSON string

# Import the json module.
# Use json.loads() to parse the JSON string into a Python dictionary.


# Step 2: Access the nested “salary” key

# Access the “salary” key using nested dictionary access (e.g., data["company"]["employee"]["payable"]["salary"]).
# Print the value of the “salary” key.


# Step 3: Add the “birth_date” key

# Add a new key-value pair to the “employee” dictionary: "birth_date": "YYYY-MM-DD".
# Replace "YYYY-MM-DD" with an actual date.


# Step 4: Save the JSON to a file

# Open a file in write mode ("w").
# Use json.dump() to write the modified dictionary to the file in JSON format.
# Use the indent parameter to make the JSON file more readable.

import json

sampleJson = """{ 
   "company":{ 
      "employee":{ 
         "name":"emma",
         "payable":{ 
            "salary":7000,
            "bonus":800
         }
      }
   }
}"""

data = json.loads(sampleJson)

print(data["company"]["employee"]["payable"]["salary"])

# Add the birth_date key to the employee dictionary
data["company"]["employee"]["birth_date"] = "1990-01-01"

print(data["company"]["employee"]["birth_date"])

with open("modified_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)
