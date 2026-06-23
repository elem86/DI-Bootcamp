# Instructions:
# Tic Tac Toe is played on a 3x3 grid. Players take turns marking empty squares with their symbol (‘O’ or ‘X’). The first player to get three of their symbols in a row (horizontally, vertically, or diagonally) wins. If all squares are filled and no player has three in a row, the game is a tie.


# Step 1: Representing the Game Board

# You’ll need a way to represent the 3x3 grid.
# A list of lists (a 2D list) is a good choice.
# Initially, each cell in the grid should be empty (e.g., represented by a space ‘ ‘).


# Step 2: Displaying the Game Board

# Create a function called display_board() to print the current state of the game board.
# The output should visually represent the 3x3 grid.
# Think about how to format the output to make it easy to read.


# Step 3: Getting Player Input

# Create a function called player_input(player) to get the player’s move.
# The function should ask the player to enter a position (e.g., row and column numbers).
# Validate the input to ensure it’s within the valid range and that the chosen cell is empty.
# Think about how to ask the user for input, and how to validate that input.


# Step 4: Checking for a Winner

# Create a function called check_win(board, player) to check if the current player has won.
# The function should check all possible winning combinations (rows, columns, diagonals).
# If a player has won, return True; otherwise, return False.
# Think about how to check every possible winning combination.


# Step 5: Checking for a Tie

# Create a function to check if the game has resulted in a tie.
# The function should check if all positions of the board are full, without a winner.


# Step 6: The Main Game Loop

# Create a function called play() to manage the game flow.
# Initialize the game board.
# Use a while loop to continue the game until there’s a winner or a tie.
# Inside the loop:
# Display the board.
# Get the current player’s input.
# Update the board with the player’s move.
# Check for a winner.
# Check for a tie.
# Switch to the next player.
# After the loop ends, display the final result (winner or tie).


# Tips:

# Consider creating helper functions to break down the logic into smaller, manageable parts.
# Follow the single responsibility principle: each function should do one thing and do it well.
# Think about how to switch between players.
# Think about how you will store the player’s symbol.


def create_board():
    # Create and return an empty 3x3 Tic Tac Toe board.
    return [
        [" ", " ", " "],
        [" ", " ", " "],
        [" ", " ", " "],
    ]


def display_board(board):
    # Print the board in a readable grid format.
    print("-------------")
    for row in board:
        print(f"| {row[0]} | {row[1]} | {row[2]} |")
        print("-------------")


def player_input(board, player):
    # Ask the current player to choose a row and column.
    # Keep asking until valid input is provided.
    while True:
        try:
            row = int(input(f"Player {player}, enter row (1-3): "))
            col = int(input(f"Player {player}, enter column (1-3): "))
        except ValueError:
            print("Please enter numbers only.")
            continue

        if row < 1 or row > 3 or col < 1 or col > 3:
            print("Row and column must be between 1 and 3.")
            continue

        row_index = row - 1
        col_index = col - 1

        if board[row_index][col_index] != " ":
            print("That cell is already taken. Choose another one.")
            continue

        board[row_index][col_index] = player
        break


def check_win(board, player):
    # Check every possible winning combination.
    # Check each row.
    for row in board:
        if row[0] == player and row[1] == player and row[2] == player:
            return True

    # Check each column.
    for col in range(3):
        if (
            board[0][col] == player
            and board[1][col] == player
            and board[2][col] == player
        ):
            return True

    # Check the two diagonals.
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True

    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True

    return False


def check_tie(board):
    # Return True only if all cells are filled and no winner exists.
    for row in board:
        for cell in row:
            if cell == " ":
                return False

    return True


def switch_player(current_player):
    # Alternate between the two players X and O.
    if current_player == "X":
        return "O"
    return "X"


def play():
    # Main game loop.
    board = create_board()
    current_player = "X"

    while True:
        display_board(board)
        player_input(board, current_player)

        if check_win(board, current_player):
            display_board(board)
            print(f"Player {current_player} wins!")
            break

        if check_tie(board):
            display_board(board)
            print("It's a tie!")
            break

        current_player = switch_player(current_player)


play()
