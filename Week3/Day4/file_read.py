with open("words.txt", "r") as f:
    content = f.read()

words = content.split()

print(f"First 10: {words[:10]}")

print(repr(content)) # The raw strong has \n characters
