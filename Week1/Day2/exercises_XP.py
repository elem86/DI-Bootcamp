# Exercise 1: Hello World
# Instructions
# Print the following output using one line of code:

# Hello world
# Hello world
# Hello world
# Hello world


print("Hello world\n" * 4)


# Exercise 2: Some Math
# Instructions
# Write code that calculates the result of:

# (99^3)*8 (meaning 99 to the power of 3, times 8).

print((99**3) * 8)


# Exercise 3: What is the output?
# Instructions
# Predict the output of the following code snippets:
# Comment what is your guess, then run the code and compare

# >>> 5 < 3 --- False
# >>> 3 == 3 --- True
# >>> 3 == "3" --- False
# >>> "3" > 3 --- TypeError
# >>> "Hello" == "hello" --- False


print(5 < 3)
print(3 == 3)
print(3 == "3")
print("Hello" == "hello")
# print("3" > 3) --- Commented out to avoid error

# 🌟 Exercise 4: Your computer brand
# Instructions
# Create a variable called computer_brand which value is the brand name of your computer.
# Using the computer_brand variable, print a sentence that states the following:
# "I have a <computer_brand> computer."

computer_brand = "Asus"
print(f"I have an {computer_brand} computer.")


# 🌟 Exercise 5: Your information
# Instructions
# Create a variable called name, and set it’s value to your name.
# Create a variable called age, and set it’s value to your age.
# Create a variable called shoe_size, and set it’s value to your shoe size.
# Create a variable called info and set it’s value to an interesting sentence about yourself. The sentence must contain all the variables created in parts 1, 2, and 3.
# Have your code print the info message.
# Run your code.

name = "Norbert"
age = 40
shoe_size = 45
info = f"My name is {name}, I am {age} years old and my shoe size is {shoe_size}. I love animals and sports."
print(info)


# 🌟 Exercise 6: A & B
# Instructions
# Create two variables, a and b.
# Each variable’s value should be a number.
# If a is bigger than b, have your code print "Hello World".

a = 10
b = 5

if a > b:
    print("Hello World")


# Exercise 7: Odd or Even
# Instructions
# Write code that asks the user for a number and determines whether this number is odd or even.
number = int(input("Enter a number: "))
if number % 2 == 0:
    print("The number is even.")
else:
    print("The number is odd.")


# 🌟 Exercise 8: What’s your name?
# Instructions
# Write code that asks the user for their name and determines whether or not you have the same name. Print out a funny message based on the outcome.

name = input("What is your name? ")
if name == "Norbert":
    print("Wow, we have the same name! That's awesome!")
else:
    print("Nice to meet you!")

#     🌟 Exercise 9: Tall enough to ride a roller coaster
# Instructions
# Write code that will ask the user for their height in centimeters.
# If they are over 145 cm, print a message that states they are tall enough to ride.
# If they are not tall enough, print a message that says they need to grow some more to ride.
height = int(input("Enter your height in centimeters: "))
if height > 145:
    print("You are tall enough to ride the roller coaster!")
else:
    print("You need to grow some more to ride the roller coaster!")
