sample_dict = {"name": "Kelly", "age": 25, "salary": 8000, "city": "New york"}
keys_to_remove = ["name", "salary"]

for key in keys_to_remove:
    del sample_dict[key]

print(sample_dict)
