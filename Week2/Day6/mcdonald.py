# Instructions: Old MacDonald’s Farm

# You are given example code and output. Your task is to create a Farm class that produces the same output.


# Step 1: Create the Farm Class

# Create a class called Farm.
# This class will represent a farm and its animals.


# Step 2: Implement the __init__ Method

# The Farm class should have an __init__ method.
# It should take one parameter: farm_name.
# Inside __init__, create two attributes: name to store the farm’s name and animals to store the animals (initialize as an empty dictionary).


# Step 3: Implement the add_animal Method

# Create a method called add_animal.
# It should take two parameters: animal_type and count (with a default value of 1). Count is the quantity of the animal that will be added to the animal dictionary.
# The dictionary will look like this:
# {'cow': 1, 'pig':3, 'horse': 2}
# If the animal_type already exists in the animals dictionary, increment its count by count.
# If it doesn’t exist, add it to the dictionary as the key and with the given count as value.


# Step 4: Implement the get_info Method

# Create a method called get_info.
# It should return a string that displays the farm’s name, the animals and their counts, and the “E-I-E-I-0!” phrase.
# Format the output to match the provided example.
# Use string formatting to align the animal names and counts into columns.


# Step 5: Test Your Code

# Create a Farm object and call the add_animal and get_info methods.
# Verify that the output matches the provided example.


class Farm:
    def __init__(self, farm_name):
        self.name = farm_name
        self.animals = {}

    def add_animal(self, animal_type=None, count=1, **kwargs):
        if animal_type is not None:
            if animal_type in self.animals:
                self.animals[animal_type] += count
            else:
                self.animals[animal_type] = count

        for animal, qty in kwargs.items():
            if animal in self.animals:
                self.animals[animal] += qty
            else:
                self.animals[animal] = qty

    def get_info(self):
        lines = [f"{self.name}'s farm", ""]
        for animal, qty in self.animals.items():
            lines.append(f"{animal} : {qty}")
        lines.append("")
        lines.append("    E-I-E-I-0!")
        return "\n".join(lines)

    def get_animal_types(self):
        return sorted(self.animals.keys())

    def get_short_info(self):
        animal_types = self.get_animal_types()
        display_names = [
            f"{animal}s" if self.animals[animal] > 1 else animal
            for animal in animal_types
        ]
        if not display_names:
            return f"{self.name}'s farm has no animals."
        if len(display_names) == 1:
            animals_str = display_names[0]
        else:
            animals_str = ", ".join(display_names[:-1]) + f" and {display_names[-1]}"
        return f"{self.name}'s farm has {animals_str}."


if __name__ == "__main__":
    macdonald = Farm("McDonald")
    macdonald.add_animal("cow", 5)
    macdonald.add_animal("sheep")
    macdonald.add_animal("sheep")
    macdonald.add_animal("goat", 12)
    print(macdonald.get_info())

    # Bonus checks
    print(macdonald.get_animal_types())
    print(macdonald.get_short_info())
