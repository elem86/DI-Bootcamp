# Challenge 1: Sorting

# Get a string of words from the user.
# Example input: without,hello,bag,world
words_string = input("Enter words separated by commas: ")

# Split the string into a list.
words_list = words_string.split(",")

# Sort the list alphabetically.
words_list.sort()

# Join the sorted list back into one string, separated by commas.
sorted_words = ",".join(words_list)

# Print the final result.
print(sorted_words)


# Challenge 2: Longest Word


def longest_word(sentence):
    # Split the sentence into a list of words.
    words = sentence.split()

    # Start by saying the first word is the longest.
    longest = words[0]

    # Check each word in the sentence.
    for word in words:
        # If the current word is longer, save it as the new longest word.
        if len(word) > len(longest):
            longest = word

    # Return the longest word found.
    return longest


print(longest_word("Margaret's toy is a pretty doll."))
print(longest_word("A thing of beauty is a joy forever."))
print(longest_word("Forgetfulness is by all means powerless!"))
