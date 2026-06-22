class Animal:
    def __init__(self, name):
        self.name = name


class Dog(Animal):
    def bark(self):
        print(f"{self.name} barked, WAF !")


# Create a Dog instance and call bark
# dog = Dog("Rex")
# dog.bark()


# class Circle:
#     color = "red"


# class NewCircle(Circle):
#     color = "blue"


# nc = NewCircle
# print(nc.color)


class Circle:
    def __init__(self, diameter):
        self.diameter = diameter

    def grow(self, factor=2):
        """grows the circle's diameter by factor"""
        self.diameter = self.diameter * factor


class NewCircle(Circle):
    def grow(self, factor=2):
        """grows the area by factor..."""
        self.diameter = self.diameter * factor * 2


nc = NewCircle(1)
print(nc.diameter)

nc.grow()

print(nc.diameter)
