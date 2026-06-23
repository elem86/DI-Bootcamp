# Instructions:

# You are given a “Matrix” string:


# MATRIX_STR = '''
# 7ir
# Tsi
# h%x
# i ?
# sM#
# $a
# #t%'''


# This represents a grid of characters, and your task is to decode the hidden message within.


# Understanding the Matrix:

# Imagine this string arranged in rows and columns, forming a grid.
# To work with it in Python, you’ll need to transform this string into a 2D list (a list of lists), where each inner list represents a row.


# Step 1: Transforming the String into a 2D List


# Step 2: Processing Columns

# Neo reads the matrix column by column, from top to bottom, starting from the leftmost column.
# You’ll need to write code that iterates through the columns of your 2D list.
# Think about how you can access the elements of a 2D list by column.


# Step 3: Filtering Alpha Characters

# only select alpha characters (letters).
# For each character in a column, check if it’s an alpha character.
# If it is, add it to a temporary string.
# Think about how you can check if a character is an alphabet letter.


# Step 4: Replacing Symbols with Spaces

# Replace every group of symbols (non-alpha characters) between two alpha characters with a space.
# After you have gathered the alpha characters, you will need to iterate through them, and where there are non alpha characters between them, you will insert a space.
# Think about how you can keep track of when you encounter an alphabet character, and when you encounter a non alphabet character.


# Step 5: Constructing the Secret Message

# Combine the filtered and processed characters to form the decoded message.
# Print the decoded message.


# Example:


# MATRIX_STR = '''
# 7ii
# Tsx
# h%?
# i #
# sM
# $a
# #t%'''

# # Step 1: Convert matrix_string to a 2D list (matrix)
# matrix = []
# # ... code to create matrix ...

# # Step 2: Iterate through columns
# # ... code to iterate through columns ...

# # Step 3: Filter alpha characters
# # ... code to filter alpha characters ...

# # Step 4: Replace symbols with spaces
# decoded_message = ""
# # ... code to replace symbols with spaces ...

# # Step 5: Print the decoded message
# print(decoded_message)

MATRIX_STR = """
7ir
Tsi
h%x
i ?
sM#
$a
#t%"""

# Step 1: Convert the multiline string into a list of rows.
# We strip leading/trailing whitespace and split at newlines.
rows = MATRIX_STR.strip().split("\n")

# Determine how many columns the grid should have: it's the longest row length.
column_count = max(len(row) for row in rows)

# Step 2: Build a rectangular 2D matrix (list of lists).
# Shorter rows are padded with spaces using ljust so every row has column_count chars.
matrix = []
for row in rows:
    matrix.append(list(row.ljust(column_count)))

# Step 3: Read characters column-by-column (top to bottom, left to right).
# Append each visited character into encoded_chars in that reading order.
encoded_chars = []
for col in range(column_count):
    for row in range(len(matrix)):
        encoded_chars.append(matrix[row][col])

# Join into a single string to simplify further processing.
encoded_message = "".join(encoded_chars)

# Step 4: Process the encoded_message to build the final decoded message.
# We want to keep alphabetic characters and replace groups of non-alpha symbols
# that occur between letters with a single space.
decoded_chars = []
inside_symbol_group = False

for char in encoded_message:
    if char.isalpha():
        # If we just passed a group of non-letter symbols and we've already
        # collected at least one decoded character, insert a space before
        # adding the next letter.
        if inside_symbol_group and decoded_chars:
            decoded_chars.append(" ")

        decoded_chars.append(char)
        inside_symbol_group = False
    else:
        # Mark that we are inside a run of non-letter symbols.
        inside_symbol_group = True

# Step 5: Join the decoded characters and print the final secret message.
decoded_message = "".join(decoded_chars)

print(decoded_message)
