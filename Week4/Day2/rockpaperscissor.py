from Week4.Day2.game import Game


def get_user_menu_choice() -> str:
    print("\nRock-Paper-Scissors")
    print("1) Play a new game")
    print("2) Show current scores")
    print("x) Quit")
    return input("Choose an option: ").strip().lower()


def print_results(results: dict):
    print("\nGame summary:")
    print(f"Wins: {results.get('win', 0)}")
    print(f"Losses: {results.get('loss', 0)}")
    print(f"Draws: {results.get('draw', 0)}")
    print("Thanks for playing!")


def main():
    results = {"win": 0, "loss": 0, "draw": 0}

    while True:
        choice = get_user_menu_choice()
        if choice == "1":
            game = Game()
            result = game.play()
            results[result] += 1
        elif choice == "2":
            print_results(results)
        elif choice in {"x", "q", "quit", "exit"}:
            print_results(results)
            break
        else:
            print("Please choose 1, 2 or x to quit.")


if __name__ == "__main__":
    main()
