# Exercise 1: Formula
# Instructions
# Write a program that calculates and prints a value according to this given formula:
# Q = Square root of [(2 * C * D)/H]
# Following are the fixed values of C and H:
# C is 50.
# H is 30.
# Ask the user for a comma-separated string of numbers, use each number from the user as D in the formula and return all the results
# For example, if the user inputs: 100,150,180
# The output should be:

# 18,22,24


# import math

# C = 50
# H = 30

# D = int(input("Enter the value of D: "))
# Q = math.sqrt((2 * C * D) / H)

# Q = round(Q)

# print(f"The value of Q is: {Q}")

# Exercise 2 : List of integers
# Instructions
# Given a list of 10 integers to analyze. For example:

#     [3, 47, 99, -80, 22, 97, 54, -23, 5, 7]
#     [44, 91, 8, 24, -6, 0, 56, 8, 100, 2]
#     [3, 21, 76, 53, 9, -82, -3, 49, 1, 76]
#     [18, 19, 2, 56, 33, 17, 41, -63, -82, 1]


# 1. Store the list of numbers in a variable.

# 2. Print the following information:
# a. The list of numbers – printed in a single line
# b. The list of numbers – sorted in descending order (largest to smallest)
# c. The sum of all the numbers

# 3. A list containing the first and the last numbers.

# 4. A list of all the numbers greater than 50.

# 5. A list of all the numbers smaller than 10.

# 6. A list of all the numbers squared – eg. for [1, 2, 3] you would print “1 4 9”.

# 7. The numbers without any duplicates – also print how many numbers are in the new list.

# 8. The average of all the numbers.

# 9. The largest number.

# 10.The smallest number.

# 11. Bonus: Find the sum, average, largest and smallest number without using built in functions.

# 12. Bonus: Instead of using pre-defined lists of numbers, ask the user for 10 numbers between -100 and 100. Ask the user for an integer between -100 and 100 – repeat this question 10 times. Each number should be added into a variable that you created earlier.

# 13. Bonus: Instead of asking the user for 10 integers, generate 10 random integers yourself. Make sure that these random integers are between -100 and 100.

# 14. Bonus: Instead of always generating 10 integers, let the amount of integers also be random! Generate a random positive integer no smaller than 50.

# 15. Bonus: Will the code work when the number of random numbers is not equal to 10?


