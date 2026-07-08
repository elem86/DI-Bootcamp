import math
import turtle
from functools import total_ordering


@total_ordering
class Circle:
    def __init__(self, radius):
        if radius <= 0:
            raise ValueError("Radius must be greater than zero.")
        self._radius = float(radius)

    @classmethod
    def from_diameter(cls, diameter):
        if diameter <= 0:
            raise ValueError("Diameter must be greater than zero.")
        return cls(diameter / 2)

    @property
    def radius(self):
        return self._radius

    @property
    def diameter(self):
        return self._radius * 2

    @property
    def area(self):
        return math.pi * self._radius**2

    def __str__(self):
        return f"Circle(radius={self.radius:.2f}, diameter={self.diameter:.2f})"

    def __repr__(self):
        return f"Circle(radius={self.radius:.2f})"

    def __add__(self, other):
        if isinstance(other, Circle):
            return Circle(self.radius + other.radius)
        if isinstance(other, (int, float)):
            return Circle(self.radius + other)
        return NotImplemented

    def __gt__(self, other):
        if not isinstance(other, Circle):
            return NotImplemented
        return self.radius > other.radius

    def __eq__(self, other):
        if not isinstance(other, Circle):
            return NotImplemented
        return math.isclose(self.radius, other.radius)

    def draw(self, x=0, y=0):
        pen = turtle.Turtle()
        pen.hideturtle()
        pen.penup()
        pen.goto(x, y - self.radius)
        pen.pendown()
        pen.circle(self.radius)
        pen.penup()


if __name__ == "__main__":
    c1 = Circle(16)
    c2 = Circle.from_diameter(64)
    c3 = Circle(24)

    print(c1)
    print(f"Area: {c1.area:.2f}")
    print(f"c2 diameter: {c2.diameter}")
    print(f"c1 > c3? {c1 > c3}")
    print(f"c2 == Circle(4)? {c2 == Circle(4)}")

    c4 = c1 + c3
    print(f"c1 + c3 = {c4}")

    circles = [c2, c1, c3, c4]
    print("Sorted:", sorted(circles))

    # Optional drawing step
    screen = turtle.Screen()
    screen.title("Sorted Circles")
    start_x = -150
    for circle in sorted(circles):
        circle.draw(start_x, 0)
        start_x += circle.diameter + 20
    screen.exitonclick()

