# 🌟 Exercise 1: Favorite Numbers
# Key Python Topics:

# Sets
# Adding/removing items in a set
# Set concatenation (using union)


# Instructions:

# Create a set called my_fav_numbers and populate it with your favorite numbers.
# Add two new numbers to the set.
# Remove the last number you added to the set.
# Create another set called friend_fav_numbers and populate it with your friend’s favorite numbers.
# Concatenate my_fav_numbers and friend_fav_numbers to create a new set called our_fav_numbers.
# Note: Sets are unordered collections, so ensure no duplicate numbers are added.

my_fav_numbers = {68, 86, 8, 10, 6}

my_fav_numbers.add(3)
my_fav_numbers.add(2)

my_fav_numbers.remove(2)

friend_fav_numbers = {1, 2, 3, 4, 8}

our_fav_numbers = my_fav_numbers.union(friend_fav_numbers)

print(our_fav_numbers)

tuple1 = (1, 2, 3, 4, 5)

# tuple1.append(6)  # This will raise an AttributeError since tuples are immutable

# print(tuple1)

tuple1 = (
    tuple1 + (6, 7, 8)
)  # This creates a new tuple by concatenating the existing tuple with a new one containing the number 6
print(tuple1)
