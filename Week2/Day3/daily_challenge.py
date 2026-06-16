# Instructions
# Here is a python code that generates a list of 20000 random numbers, called list_of_numbers, and a target number.

# import random

# list_of_numbers = [random.randint(0, 10000) for _ in range(20000)]

# target_number   = 3728


# Copy this code, and create a program that finds, within list_of_numbers all the pairs of number that sum to the target number

# For example

# 1000 and 2728 sums to the target_number 3728
# 1864 and 1864 sums to the target_number 3728


import random


list_of_numbers = [random.randint(0, 10000) for _ in range(20000)]

target_number = 3728

number_counts = {}

for number in list_of_numbers:
    if number in number_counts:
        number_counts[number] += 1
    else:
        number_counts[number] = 1

for number in number_counts:
    matching_number = target_number - number

    if matching_number not in number_counts:
        continue

    if number > matching_number:
        continue

    if number == matching_number and number_counts[number] < 2:
        continue

    print(f"{number} and {matching_number} sums to the target number {target_number}")
