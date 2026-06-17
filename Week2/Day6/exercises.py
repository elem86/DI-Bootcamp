# 🌟 Exercise 1: Cats
# Key Python Topics:

# Classes and objects
# Object instantiation
# Attributes
# Functions


# Instructions:

# Use the provided Cat class to create three cat objects. Then, create a function to find the oldest cat and print its details.


# Step 1: Create Cat Objects

# Use the Cat class to create three cat objects with different names and ages.


# Step 2: Create a Function to Find the Oldest Cat

# Create a function that takes the three cat objects as input.
# Inside the function, compare the ages of the cats to find the oldest one.
# Return the oldest cat object.


# Step 3: Print the Oldest Cat’s Details

# Call the function to get the oldest cat.
# Print a formatted string: “The oldest cat is <cat_name>, and is <cat_age> years old.”
# Replace <cat_name> and <cat_age> with the oldest cat’s name and age.


# Example:

# class Cat:
#     def __init__(self, cat_name, cat_age):
#         self.name = cat_name
#         self.age = cat_age

# # Step 1: Create cat objects
# # cat1 = create the object

# # Step 2: Create a function to find the oldest cat
# def find_oldest_cat(cat1, cat2, cat3):
#     # ... code to find and return the oldest cat ...

# # Step 3: Print the oldest cat's details


# 🌟 Exercise 2 : Dogs
# Goal: Create a Dog class, instantiate objects, call methods, and compare dog sizes.


# Key Python Topics:

# Classes and objects
# Object instantiation
# Methods
# Attributes
# Conditional statements (if)


# Instructions:

# Create a Dog class with methods for barking and jumping. Instantiate dog objects, call their methods, and compare their sizes.


# Step 1: Create the Dog Class

# Create a class called Dog.
# In the __init__ method, take name and height as parameters and create corresponding attributes.
# Create a bark() method that prints “<dog_name> goes woof!”.
# Create a jump() method that prints “<dog_name> jumps <x> cm high!”, where x is height * 2.


# Step 2: Create Dog Objects

# Create davids_dog and sarahs_dog objects with their respective names and heights.


# Step 3: Print Dog Details and Call Methods

# Print the name and height of each dog.
# Call the bark() and jump() methods for each dog.


# Step 4: Compare Dog Sizes


# 🌟 Exercise 3 : Who’s the song producer?
# Goal: Create a Song class to represent song lyrics and print them.


# Key Python Topics:

# Classes and objects
# Object instantiation
# Methods
# Lists


# Instructions:

# Create a Song class with a method to print song lyrics line by line.


# Step 1: Create the Song Class

# Create a class called Song.
# In the __init__ method, take lyrics (a list) as a parameter and create a corresponding attribute.
# Create a sing_me_a_song() method that prints each element of the lyrics list on a new line.


# Example:

# stairway = Song(["There’s a lady who's sure", "all that glitters is gold", "and she’s buying a stairway to heaven"])

# stairway.sing_me_a_song()

# Output: There’s a lady who’s sureall that glitters is goldand she’s buying a stairway to heaven


# 🌟 Exercise 4 : Afternoon at the Zoo
# Goal:

# Create a Zoo class to manage animals. The class should allow adding animals, displaying them, selling them, and organizing them into alphabetical groups.


# Key Python Topics:

# Classes and objects
# Object instantiation
# Methods
# Lists
# Dictionaries (for grouping)
# String manipulation


# Instructions
# Step 1: Define the Zoo Class
# 1. Create a class called Zoo.

# 2. Implement the __init__() method:

# It takes a string parameter zoo_name, representing the name of the zoo.
# Initialize an empty list called animals to keep track of animal names.
# 3. Add a method add_animal(new_animal):

# This method adds a new animal to the animals list.
# Do not add the animal if it is already in the list.
# 4. Add a method get_animals():

# This method prints all animals currently in the zoo.
# 5. Add a method sell_animal(animal_sold):

# This method checks if a specified animal exists on the animals list and if so, remove from it.
# 6. Add a method sort_animals():

# This method sorts the animals alphabetically.
# It also groups them by the first letter of their name.
# The result should be a dictionary where:
# Each key is a letter.
# Each value is a list of animals that start with that letter.
# Example output:

# {
#    'B': ['Baboon', 'Bear'],
#    'C': ['Cat', 'Cougar'],
#    'G': ['Giraffe'],
#    'L': ['Lion'],
#    'Z': ['Zebra']
# }
# 7. Add a method get_groups():

# This method prints the grouped animals as created by sort_animals().
# Example output:

# B: ['Baboon', 'Bear']
# C: ['Cat', 'Cougar']
# G: ['Giraffe']
# ...


# Step 2: Create a Zoo Object
# Create an instance of the Zoo class and pass a name for the zoo.


# Step 3: Call the Zoo Methods
# Use the methods of your Zoo object to test adding, selling, displaying, sorting, and grouping animals.


# Example (No Internal Logic Provided)
# class Zoo:
#     def __init__(self, zoo_name):
#         pass

#     def add_animal(self, new_animal):
#         pass

#     def get_animals(self):
#         pass

#     def sell_animal(self, animal_sold):
#         pass

#     def sort_animals(self):
#         pass

#     def get_groups(self):
#         pass

# # Step 2: Create a Zoo instance
# brooklyn_safari = Zoo("Brooklyn Safari")

# # Step 3: Use the Zoo methods
# brooklyn_safari.add_animal("Giraffe")
# brooklyn_safari.add_animal("Bear")
# brooklyn_safari.add_animal("Baboon")
# brooklyn_safari.get_animals()
# brooklyn_safari.sell_animal("Bear")
# brooklyn_safari.get_animals()
# brooklyn_safari.sort_animals()
# brooklyn_safari.get_groups()


# Bonus: Modify the add_animal() method to get *args so you dont need to repeat the method each time for a new animal, you can pass multiple animals names separated by a comma.


# Exercise 1: Cats


class Cat:
    def __init__(self, cat_name, cat_age):
        self.name = cat_name
        self.age = cat_age


# Step 1: Create three cat objects.
cat1 = Cat("Milo", 3)
cat2 = Cat("Luna", 7)
cat3 = Cat("Simba", 5)


# Step 2: Create a function that finds the oldest cat.
def find_oldest_cat(cat1, cat2, cat3):
    oldest_cat = cat1

    if cat2.age > oldest_cat.age:
        oldest_cat = cat2

    if cat3.age > oldest_cat.age:
        oldest_cat = cat3

    return oldest_cat


# Step 3: Print the oldest cat's details.
oldest_cat = find_oldest_cat(cat1, cat2, cat3)
print(f"The oldest cat is {oldest_cat.name}, and is {oldest_cat.age} years old.")


# Exercise 2: Dogs


class Dog:
    def __init__(self, name, height):
        self.name = name
        self.height = height

    def bark(self):
        print(f"{self.name} goes woof!")

    def jump(self):
        jump_height = self.height * 2
        print(f"{self.name} jumps {jump_height} cm high!")


# Step 2: Create dog objects.
davids_dog = Dog("Rex", 50)
sarahs_dog = Dog("Teacup", 20)


# Step 3: Print dog details and call their methods.
print(f"David's dog is named {davids_dog.name} and is {davids_dog.height} cm tall.")
davids_dog.bark()
davids_dog.jump()

print(f"Sarah's dog is named {sarahs_dog.name} and is {sarahs_dog.height} cm tall.")
sarahs_dog.bark()
sarahs_dog.jump()


# Step 4: Compare dog sizes.
if davids_dog.height > sarahs_dog.height:
    print(f"{davids_dog.name} is bigger than {sarahs_dog.name}.")
elif sarahs_dog.height > davids_dog.height:
    print(f"{sarahs_dog.name} is bigger than {davids_dog.name}.")
else:
    print(f"{davids_dog.name} and {sarahs_dog.name} are the same height.")

# Exercise 3: Who's the song producer?


class Song:
    def __init__(self, lyrics):
        self.lyrics = lyrics

    def sing_me_a_song(self):
        for line in self.lyrics:
            print(line)


stairway = Song(
    [
        "There's a lady who's sure",
        "all that glitters is gold",
        "and she's buying a stairway to heaven",
    ]
)

stairway.sing_me_a_song()


# Exercise 4: Afternoon at the Zoo


class Zoo:
    def __init__(self, zoo_name):
        self.name = zoo_name
        self.animals = []
        self.grouped_animals = {}

    def add_animal(self, *new_animals):
        for animal in new_animals:
            if animal not in self.animals:
                self.animals.append(animal)

    def get_animals(self):
        print(self.animals)

    def sell_animal(self, animal_sold):
        if animal_sold in self.animals:
            self.animals.remove(animal_sold)

    def sort_animals(self):
        sorted_animals = sorted(self.animals)
        grouped_animals = {}

        for animal in sorted_animals:
            first_letter = animal[0]

            if first_letter not in grouped_animals:
                grouped_animals[first_letter] = []

            grouped_animals[first_letter].append(animal)

        self.grouped_animals = grouped_animals
        return grouped_animals

    def get_groups(self):
        for letter, animals in self.grouped_animals.items():
            print(f"{letter}: {animals}")


brooklyn_safari = Zoo("Brooklyn Safari")

brooklyn_safari.add_animal(
    "Giraffe", "Bear", "Baboon", "Lion", "Zebra", "Cat", "Cougar"
)
brooklyn_safari.get_animals()

brooklyn_safari.sell_animal("Bear")
brooklyn_safari.get_animals()

brooklyn_safari.sort_animals()
brooklyn_safari.get_groups()
