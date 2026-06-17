# What are the results?

my_string = "Hello Class"
my_number = 18
# Exercise 1
print(my_string + my_string * 3)


# Exercise 2
print((my_string + '\n')*2, end='!!!\n')
# Hello Class
# Hello Class
# !!!

# Exercise 3
print(my_string == my_string.lower())
# hello class

# Exercise 4
print(my_number // 5)
# 3

# Exercise 5
print(my_string + "_" * (my_number // 9) + str(my_number))
# Typerror

# Exercise 6
print(my_number % len(my_string))
# 1

# Exercise 7
print(my_string[8].capitalize())
# A

# Exercise 8
if my_number % 4 != 0:
    print(my_string[0:4])
else:
    print(my_string[-3])
# 

# Exercise 9
print(bool(len(my_string) > 5))
# True

# Exercise 10
print(bool(my_number < 5) + 7)
# 7

# Exercise 11
print(abs(bool(my_number > 10) - 8))
# 8

# Exercise 12
print(not bool(None))


# Exercise 13
print(int(my_number is str))
# Exercise 14
print(str(3 == 3 == 3))
# Exercise 15
print(float(3 == 3 == 3))
# Exercise 16
print(my_string[my_number // len(my_string)])
# Exercise 17
print((' ' not in my_string) * 3.5)
# Exercise 18
print(f"{len(my_string)} chars in \"{my_string}\"")
# Exercise 19 (*)
print(ord(my_string[3]))
# Exercise 20 (*)
result = ""
for c in my_string:
    if 96 < ord(c) < 123:
        result += c
print(result)