# 🌟 Exercise 1: Currencies
# Goal: Implement dunder methods for a Currency class to handle string representation, integer conversion, addition, and in-place addition.


# Key Python Topics:

# Dunder methods (__str__, __repr__, __int__, __add__, __iadd__)
# Type checking (isinstance())
# Raising exceptions (raise TypeError)


# Instructions:

# class Currency:
#     def __init__(self, currency, amount):
#         self.currency = currency
#         self.amount = amount

#     #Your code starts HERE


# Using the code above, implement the relevant methods and dunder methods which will output the results below.

# Hint : When adding 2 currencies which don’t share the same label you should raise an error.

# c1 = Currency('dollar', 5)
# c2 = Currency('dollar', 10)
# c3 = Currency('shekel', 1)
# c4 = Currency('shekel', 10)

# #the comment is the expected output
# print(c1)
# # '5 dollars'

# print(int(c1))
# # 5

# print(repr(c1))
# # '5 dollars'

# print(c1 + 5)
# # 10

# print(c1 + c2)
# # 15

# print(c1)
# # 5 dollars

# c1 += 5
# print(c1)
# # 10 dollars

# c1 += c2
# print(c1)
# # 20 dollars

# print(c1 + c3)
# # TypeError: Cannot add between Currency type <dollar> and <shekel>
# #comment the print above before you run the file for next exercises (since the error will cra


class Currency:
    def __init__(self, currency, amount):
        self.currency = currency
        self.amount = amount

    def __str__(self):
        name = self.currency
        amt = self.amount
        if amt == 1:
            return f"{amt} {name}"
        return f"{amt} {name}s"

    def __repr__(self):
        return self.__str__()

    def __int__(self):
        return int(self.amount)

    def _check_currency_match(self, other):
        if self.currency != other.currency:
            raise TypeError(
                f"Cannot add between Currency type <{self.currency}> and <{other.currency}>"
            )

    def __add__(self, other):
        if isinstance(other, Currency):
            self._check_currency_match(other)
            return self.amount + other.amount
        if isinstance(other, (int, float)):
            return self.amount + other
        return NotImplemented

    def __radd__(self, other):
        # support int + Currency
        if isinstance(other, (int, float)):
            return other + self.amount
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, Currency):
            self._check_currency_match(other)
            self.amount += other.amount
            return self
        if isinstance(other, (int, float)):
            self.amount += other
            return self
        return NotImplemented


c1 = Currency("dollar", 5)
c2 = Currency("dollar", 10)
c3 = Currency("shekel", 1)
c4 = Currency("shekel", 10)

# the comment is the expected output
print(c1)
# '5 dollars'

print(int(c1))
# 5

print(repr(c1))
# '5 dollars'

print(c1 + 5)
# 10

print(c1 + c2)
# 15

print(c1)
# 5 dollars

c1 += 5
print(c1)
# 10 dollars

c1 += c2
print(c1)
# 20 dollars

# print(c1 + c3)
# TypeError: Cannot add between Currency type <dollar> and <shekel>
# comment the print above before you run the file for next exercises (since the error will crash your file)


# 🌟 Exercise 3: String module
# Goal: Generate a random string of length 5 using the string module.


# Instructions:

# Use the string module to generate a random string of length 5, consisting of uppercase and lowercase letters only.


# Key Python Topics:

# string module
# random module
# String concatenation


# Step 1: Import the string and random modules

# Import the string and random modules.


# Step 2: Create a string of all letters

# Read about the strings methods HERE to find the best methods for this step


# Step 3: Generate a random string

# Use a loop to select 5 random characters from the combined string.
# Concatenate the characters to form the random string.

import string
import random

all_letters = string.ascii_letters
# allows repeats
rand = "".join(random.choices(all_letters, k=5))


print(rand)


# 🌟 Exercise 4: Current Date
# Goal: Create a function that displays the current date.


# Key Python Topics:

# datetime module


# Instructions:

# Use the datetime module to create a function that displays the current date.

# Step 1: Import the datetime module

# Step 2: Get the current date

# Step 3: Display the date


from datetime import date, datetime

current_date = date.today()
print(current_date)


# 🌟 Exercise 5: Amount of time left until January 1st
# Goal: Create a function that displays the amount of time left until January 1st.


# Key Python Topics:

# datetime module
# Time difference calculations


# Instructions:

# Use the datetime module to calculate and display the time left until January 1st.
# more info about this module HERE

# Step 1: Import the datetime module

# Step 2: Get the current date and time

# Step 3: Create a datetime object for January 1st of the next year

# Step 4: Calculate the time difference

# Step 5: Display the time difference


date_time = datetime.now()

next_year = datetime.now().year + 1

jan_1 = datetime(next_year, 1, 1)

difference_time = jan_1 - date_time

print(difference_time)


# 🌟 Exercise 6: Birthday and minutes
# Key Python Topics:

# datetime module
# datetime.datetime.strptime() (parsing dates)
# Time difference calculations
# .total_seconds() method


# Instructions:

# Create a function that accepts a birthdate as an argument (in the format of your choice), then displays a message stating how many minutes the user lived in his life.


def birth_function(birthdate):
    birthday = datetime.strptime(birthdate, "%Y-%m-%d")
    now = datetime.now()
    minutes_lived = int((now - birthday).total_seconds() / 60)
    print(f"You have lived {minutes_lived} minutes.")


# Example usage:
birth_function("1986-03-11")


# 🌟 Exercise 7: Faker Module
# Goal: Use the faker module to generate fake user data and store it in a list of dictionaries.
# Read more about this module HERE


# Key Python Topics:

# faker module
# Dictionaries
# Lists
# Loops


# Instructions:

# Install the faker module and use it to create a list of dictionaries, where each dictionary represents a user with fake data.

# Step 1: Install the faker module

# Step 2: Import the faker module

# Step 3: Create an empty list of users

# Step 4: Create a function to add users

# Create a function that takes the number of users to generate as an argument.
# Inside the function, use a loop to generate the specified number of users.
# For each user, create a dictionary with the keys name, address, and language_code.
# Use the faker instance to generate fake data for each key:
# name: faker.name()
# address: faker.address()
# language_code: faker.language_code()
# Append the user dictionary to the users list.
# Step 5: Call the function and print the users list

from faker import Faker

fake = Faker()


def add_users(num_users):
    new_users = []
    for _ in range(num_users):
        user = {
            "name": fake.name(),
            "address": fake.address(),
            "language_code": fake.language_code(),
        }
        new_users.append(user)
    return new_users


users = add_users(5)
print(users)
