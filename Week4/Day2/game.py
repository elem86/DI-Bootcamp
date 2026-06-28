import random


class Game:
    VALID_ITEMS = ["rock", "paper", "scissors"]

    def get_user_item(self) -> str:
        while True:
            user_input = input("Choose rock, paper or scissors: ").strip().lower()
            if user_input in self.VALID_ITEMS:
                return user_input
            print("Invalid choice. Please type rock, paper or scissors.")

    def get_computer_item(self) -> str:
        return random.choice(self.VALID_ITEMS)

    def get_game_result(self, user_item: str, computer_item: str) -> str:
        if user_item == computer_item:
            return "draw"

        wins_against = {
            "rock": "scissors",
            "paper": "rock",
            "scissors": "paper",
        }

        return "win" if wins_against[user_item] == computer_item else "loss"

    def play(self) -> str:
        user_item = self.get_user_item()
        computer_item = self.get_computer_item()
        result = self.get_game_result(user_item, computer_item)

        if result == "win":
            message = "You win!"
        elif result == "loss":
            message = "You lose."
        else:
            message = "It's a draw."

        print(
            f"You selected {user_item}. The computer selected {computer_item}. {message}"
        )
        return result
