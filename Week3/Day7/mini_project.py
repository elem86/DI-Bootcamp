class Human:
    def __init__(self, id_number, name, age, priority, blood_type, family=None):
        self.id_number = str(id_number)
        self.name = str(name)
        self.age = int(age)
        self.priority = bool(priority)
        self.blood_type = str(blood_type).upper()
        self.family = []

        valid_blood_types = {"A", "B", "AB", "O"}
        if self.blood_type not in valid_blood_types:
            raise ValueError("blood_type must be one of: A, B, AB, O")

        if family is not None:
            for member in family:
                self.add_family_member(member)

    def add_family_member(self, person):
        if not isinstance(person, Human):
            raise TypeError("Only Human objects can be added to the family")

        if person is self:
            return

        if person not in self.family:
            self.family = self.family + [person]

        if self not in person.family:
            person.family = person.family + [self]


class Queue:
    def __init__(self):
        self.humans = []

    def add_person(self, person):
        if not isinstance(person, Human):
            raise TypeError("Only Human objects can be added to the queue")

        if person.age > 60 or person.priority:
            self.humans = [person] + self.humans
        else:
            self.humans = self.humans + [person]

    def find_in_queue(self, person):
        if not isinstance(person, Human):
            raise TypeError("Only Human objects can be searched in the queue")

        for index, current in enumerate(self.humans):
            if current is person:
                return index

        raise ValueError("This person is not in the queue")

    def swap(self, person1, person2):
        if person1 not in self.humans or person2 not in self.humans:
            raise ValueError("Both people must be in the queue")

        index1 = None
        index2 = None

        for index, person in enumerate(self.humans):
            if person is person1:
                index1 = index
            elif person is person2:
                index2 = index

            if index1 is not None and index2 is not None:
                break

        if index1 is None or index2 is None:
            raise ValueError("Both people must be in the queue")

        self.humans[index1], self.humans[index2] = (
            self.humans[index2],
            self.humans[index1],
        )

    def _share_family(self, person1, person2):
        if not isinstance(person1, Human) or not isinstance(person2, Human):
            return False
        return person2 in person1.family or person1 in person2.family

    def _insert_person(self, queue, person):
        if not queue:
            return [person]

        for index in range(len(queue) + 1):
            before = queue[:index]
            after = queue[index:]

            if not before:
                if not after or not self._share_family(person, after[0]):
                    return [person] + after
            elif not after:
                if not self._share_family(before[-1], person):
                    return before + [person]
            else:
                if not self._share_family(
                    before[-1], person
                ) and not self._share_family(person, after[0]):
                    return before + [person] + after

        return queue + [person]

    def rearrange_queue(self):
        if not self.humans:
            return None

        ordered = []
        for person in self.humans:
            ordered = self._insert_person(ordered, person)

        self.humans = ordered
        return self.humans

    def get_next(self):
        if not self.humans:
            return None

        next_person = self.humans[0]
        self.humans = self.humans[1:]
        return next_person

    def get_next_blood_type(self, blood_type):
        if not self.humans:
            return None

        target_type = str(blood_type).upper()

        for index, person in enumerate(self.humans):
            if person.blood_type == target_type:
                self.humans = self.humans[:index] + self.humans[index + 1 :]
                return person

        return None

    def sort_by_age(self):
        if not self.humans:
            return None

        ordered = list(self.humans)
        length = len(ordered)

        for i in range(length):
            best_index = i
            for j in range(i + 1, length):
                left = ordered[best_index]
                right = ordered[j]

                if left.priority is False and right.priority is True:
                    best_index = j
                elif left.priority == right.priority and left.age < right.age:
                    best_index = j

            if best_index != i:
                ordered[i], ordered[best_index] = ordered[best_index], ordered[i]

        self.humans = ordered
        return self.humans


if __name__ == "__main__":
    # quick demo run so the module prints something when executed
    a = Human(1, "Alice", 30, False, "A")
    b = Human(2, "Bob", 70, False, "O")
    c = Human(3, "Carol", 25, True, "B")
    d = Human(4, "Dave", 40, False, "AB")

    a.add_family_member(d)

    q = Queue()
    for p in (a, b, c, d):
        q.add_person(p)

    q.rearrange_queue()
    print("Queue order:", [person.name for person in q.humans])

    found_o = q.get_next_blood_type("O")
    print("Next with blood type O:", found_o.name if found_o else None)

    next_served = q.get_next()
    print("Serving next:", next_served.name if next_served else None)
