from datetime import date


def get_age(year, month, day):
    today = date.today()
    age = today.year - year

    if (today.month, today.day) < (month, day):
        age -= 1

    return age


def can_retire(gender, date_of_birth):
    year, month, day = date_of_birth.split("/")

    year = int(year)
    month = int(month)
    day = int(day)

    age = get_age(year, month, day)

    if gender == "m" and age >= 67:
        return True
    elif gender == "f" and age >= 62:
        return True
    else:
        return False


gender = input("Enter your gender (m/f): ")
date_of_birth = input("Enter your date of birth (yyyy/mm/dd): ")

if can_retire(gender, date_of_birth):
    print("You can retire.")
else:
    print("You cannot retire yet.")
