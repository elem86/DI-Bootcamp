try:
    with open("data.txt", "r") as f:
        content = f.read()
except FileNotFoundError:
    print("File not found! Using default data")
    content = "default"
