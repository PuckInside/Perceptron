seats = [ 0, 0, 0, 1, 0, 1]

def GetMaxEquidistantIndex(values: list) -> int:
    max_first_index = 0
    max_second_index = 0
    last_index = 0

    for x in range(len(values)):
        if values[x] != 0 or x == len(values) - 1:
            if (max_second_index - max_first_index) < (x - last_index):
                max_first_index = last_index
                max_second_index = x

            last_index = x

    if values[max_first_index] != 1:
        return max_first_index

    if values[max_second_index] != 1:
        return max_second_index

    return (max_second_index + max_first_index) // 2


free_seats = GetMaxEquidistantIndex(seats)
print(free_seats)

    