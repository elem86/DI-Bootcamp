# 🌟 Exercise 1: What Are You Learning?
# Goal: Create a function that displays a message about what you’re learning.

# Key Python Topics:

# Functions (defining and calling)
# print() function


# Step 1: Define a Function

# Define a function named display_message().
# This function should not take any parameters.


# Step 2: Print a Message

# For example: “I am learning about functions in Python.”


# Step 3: Call the Function

# This will execute the code inside the function and print your message.


# Expected Output:

# I am learning about functions in Python.


def display_message():
    print("I am learning about functions in Python.")


display_message()


# 🌟 Exercise 2: What’s Your Favorite Book?
# Goal: Create a function that displays a message about a favorite book.

# Key Python Topics:

# Functions with parameters
# String concatenation or f-strings
# Calling functions with arguments


# Step 1: Define a Function with a Parameter

# Define a function named favorite_book().
# This function should accept one parameter called title.


# Step 2: Print a Message with the Title

# The function needs to output a message like “One of my favorite books is <title>”.


# Step 3: Call the Function with an Argument

# Call the favorite_book() function and provide a book title as an argument.
# For example: favorite_book("Alice in Wonderland").


def favorite_book(title):
    print(f"One of my favorite book is {title}")


favorite_book("Chainsaw")


# 🌟 Exercise 3: Some Geography
# Goal: Create a function that describes a city and its country.

# Key Python Topics:

# Functions with multiple parameters
# Default parameter values
# String formatting


# Step 1: Define a Function with Parameters ok

# Define a function named describe_city().
# This function should accept two parameters: city and country.
# Give the country parameter a default value, such as “Unknown”.


# Step 2: Print a Message

# Inside the function, set up the code to display a sentence like “ is in “.
# Replace <city> and <country> with the parameter values.


# Step 3: Call the Function

# Call the describe_city() function with different city and country combinations.
# Try calling it with and without providing the country argument to see the default value in action.
# Example: describe_city("Reykjavik", "Iceland") and describe_city("Paris").


# Expected Output:

# Reykjavik is in Iceland.
# Paris is in Unknown.


def describe_city(city, country="Unknown"):
    print(f"{city} is in {country}")


describe_city("Paris")


# Exercise 4: Random
# Goal: Create a function that generates random numbers and compares them.

# Key Python Topics:

# random module
# random.randint() function
# Conditional statements (if, else)


# Step 1: Import the random Module

# At the beginning of your script, use import random to access the random number generation functions.


# Step 2: Define a Function with a Parameter

# Create a function that accepts a number between 1 and 100 as a parameter.


# Step 3: Generate a Random Number

# Inside the function, use random.randint(1, 100) to generate a random integer between 1 and 100.


# Step 4: Compare the Numbers

# If they are the same, print a success message. Otherwise, print a fail message and display both numbers.


# Step 5: Call the Function

# Call the function with a number between 1 and 100.


# Expected Output:

# Success! (if the numbers match)
# Fail! Your number: 50, Random number: 23 (if they don't match)


import random


def random_numbers(number1):
    number2 = random.randint(1, 100)

    if number1 == number2:
        print("Success")
    else:
        print(f"Fail! Your number: {number1}, Random number: {number2}")


user_number = int(input("Plase enter a number 1 to 100: "))

random_numbers(user_number)


# 🌟 Exercise 5: Let’s Create Some Personalized Shirts!
# Goal: Create a function to describe a shirt’s size and message, with default values.

# Key Python Topics:

# Functions with parameters and default values
# Keyword arguments


# Step 1: Define a Function with Parameters

# Define a function called make_shirt().
# This function should accept two parameters: size and text.


# Step 2: Print a Summary Message

# Set up the function to display a sentence summarizing the shirt’s size and message.


# Step 3: Call the Function


# Step 4: Modify the Function with Default Values

# Modify the make_shirt() function so that size has a default value of “large” and text has a default value of “I love Python”.


# Step 5: Call the Function with Default and Custom Values

# Call make_shirt() to make a large shirt with the default message.
# Call make_shirt() to make a medium shirt with the default message.
# Call make_shirt() to make a shirt of any size with a different message.


# Step 6 (Bonus): Keyword Arguments

# Call make_shirt() using keyword arguments (e.g., make_shirt(size="small", text="Hello!")).


# Expected Output:

# The size of the shirt is large and the text is I love Python.
# The size of the shirt is medium and the text is I love Python.
# The size of the shirt is small and the text is Custom message.


def make_shirt(size="Large", text="I love Python"):
    print(f"The size of the shirt is {size} and the text is {text}.")


# make_shirt()
make_shirt("Medium")
make_shirt("Small", "I like sports")

# 🌟 Exercise 6: Magicians…
# Goal: Modify a list of magician names and display them in different ways.

# Key Python Topics:

# Lists
# for loops
# Modifying lists
# Functions that modify data structures


# Step 1: Create a List of Magician Names

# Create a list called magician_names with the given names:
# ['Harry Houdini', 'David Blaine', 'Criss Angel']


# Step 2: Create a Function to Display Magicians

# Create a function called show_magicians() that takes the magician_names list as a parameter.
# Inside the function, iterate through the list and print each magician’s name.


# Step 3: Create a Function to Modify the List

# Create a function called make_great() that takes the magician_names list as a parameter.
# Inside the function, use a for loop to iterate through the list and add “the Great” before each magician’s name.


# Step 4: Call the Functions

# Call make_great() to modify the list.
# Call show_magicians() to display the modified list.


# Expected Output:

# Harry Houdini the Great
# David Blaine the Great
# Criss Angel the Great


magician_names = ["Harry Houdini", "David Blaine", "Criss Angel"]


def show_magicians(magician_names):
    for magician in magician_names:
        print(magician)


def make_great(magician_names):
    for index in range(len(magician_names)):
        magician_names[index] = magician_names[index] + " the Great"


make_great(magician_names)
show_magicians(magician_names)


# 🌟 Exercise 7: Temperature Advice
# Goal: Generate a random temperature and provide advice based on the temperature range.

# Key Python Topics:

# Functions
# Conditionals (if / elif)
# Random numbers
# Floating-point numbers (Bonus)
# Handling seasons (Bonus)


# Step 1: Create the get_random_temp() Function

# Create a function called get_random_temp() that returns a random integer between -10 and 40 degrees Celsius.


# Step 2: Create the main() Function

# Create a function called main(). Inside this function:
# Call get_random_temp() to get a random temperature.
# Store the temperature in a variable and print a friendly message like:
# “The temperature right now is 32 degrees Celsius.”


# Step 3: Provide Temperature-Based Advice

# Inside main(), provide advice based on the temperature:
# Below 0°C: e.g., “Brrr, that’s freezing! Wear some extra layers today.”
# Between 0°C and 16°C: e.g., “Quite chilly! Don’t forget your coat.”
# Between 16°C and 23°C: e.g., “Nice weather.”
# Between 24°C and 32°C: e.g., “A bit warm, stay hydrated.”
# Between 32°C and 40°C: e.g., “It’s really hot! Stay cool.”


# Step 4: Floating-Point Temperatures (Bonus)

# Modify get_random_temp() to return a random floating-point number using random.uniform() for more accurate temperature values.


# Step 5: Month-Based Seasons (Bonus)

# Instead of directly generating a random temperature, ask the user for a month (1-12) and determine the season using if/elif conditions.
# Modify get_random_temp() to return temperatures specific to each season.


# Expected Output:

# The temperature right now is 32 degrees Celsius.
# It's really hot! Stay cool.

import random


def get_random_temp():
    return random.uniform(-10, 40)


def main():
    get_random_temp()
    temp = get_random_temp()
    print(f"The temperature right now is {temp} degrees Celsius.")

    if temp < 0:
        print("Brrr, thats freezing! Wear some extra layers today.")
    elif temp > 0 and temp < 16:
        print("Quite chilly! Dont forget your coat.")
    elif temp > 16 and temp < 23:
        print("Nice weather.")
    elif temp > 24 and temp < 32:
        print("A bit warm, stay hydrated.")
    elif temp > 32 and temp < 40:
        print("Its really hot! Stay cool.")


main()
