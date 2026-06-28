# Part 1 : Quizz :
# Answer the following questions

# What is a class?
# A class is a blueprint or template for creating objects, defining their
# data attributes and behavior through methods.

# What is an instance?
# An instance is a concrete object created from a class, with its own data
# stored in the attributes defined by the class.

# What is encapsulation?
# Encapsulation is the practice of bundling data and methods together and
# restricting direct access to internal object state.

# What is abstraction?
# Abstraction means exposing only the essential features of an object while
# hiding the implementation details.

# What is inheritance?
# Inheritance is a mechanism where one class (child) derives from another
# class (parent), reusing or extending its behavior.

# What is multiple inheritance?
# Multiple inheritance is when a class inherits behavior from more than one
# parent class.

# What is polymorphism?
# Polymorphism allows code to use objects of different classes through the
# same interface or method name.

# What is method resolution order or MRO?
# MRO is the order Python follows to look up methods and attributes in a class
# hierarchy, especially when multiple inheritance is involved.


import random


# Part 2: Create a deck of cards class.
# The Deck of cards class should NOT inherit from a Card class.

# The requirements are as follows:

# The Card class should have a suit (Hearts, Diamonds, Clubs, Spades) and a value (A,2,3,4,5,6,7,8,9,10,J,Q,K)
# The Deck class :
# should have a shuffle method which makes sure the deck of cards has all 52 cards and then rearranges them randomly.
# should have a method called deal which deals a single card from the deck. After a card is dealt, it should be removed from the deck.


class Card:
    def __init__(self, suit: str, value: str):
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f"{self.value} of {self.suit}"


class Deck:
    SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
    VALUES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

    def __init__(self):
        self.cards = []
        self.shuffle()

    def build_deck(self):
        self.cards = [Card(suit, value) for suit in self.SUITS for value in self.VALUES]

    def shuffle(self):
        self.build_deck()
        random.shuffle(self.cards)

    def deal(self):
        if not self.cards:
            raise IndexError("No cards left to deal")
        return self.cards.pop()


def main():
    deck = Deck()
    print(f"Deck has {len(deck.cards)} cards after shuffle.")

    while True:
        if not deck.cards:
            print("No cards left in the deck.")
            break

        choice = input("Deal a card? (y/n): ").strip().lower()
        if choice not in {"y", "yes"}:
            print("Stopping dealing.")
            break

        card = deck.deal()
        print(f"Dealt card: {card}")
        print(f"Cards remaining: {len(deck.cards)}")

    print("Goodbye.")


if __name__ == "__main__":
    main()
