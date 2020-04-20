import os
import random

# Used to locate the cheese
from colorama import Fore, Style

CHEESE_POSITION_ROW = []
CHEESE_POSITION_COL = []


def is_safe(matrix, visited, i, j):
    return not (matrix[i][j] == 'X' or visited[i][j] != 0)


def is_valid(matrix, i, j):
    return 0 <= i < len(matrix) and 0 <= j < len(matrix[0])


def find_path(matrix, visited, i, j, dest_row, dest_col, min_dist, dist):
    if i == dest_row and j == dest_col:
        return min(dist, min_dist)

    visited[i][j] = 'V'

    # down
    if is_valid(matrix, i + 1, j) and is_safe(matrix, visited, i + 1, j):
        min_dist = find_path(matrix, visited, i + 1, j, dest_row, dest_col, min_dist, dist + 1)

    # right
    if is_valid(matrix, i, j + 1) and is_safe(matrix, visited, i, j + 1):
        min_dist = find_path(matrix, visited, i, j + 1, dest_row, dest_col, min_dist, dist + 1)

    # top
    if is_valid(matrix, i - 1, j) and is_safe(matrix, visited, i - 1, j):
        min_dist = find_path(matrix, visited, i - 1, j, dest_row, dest_col, min_dist, dist + 1)

    # left
    if is_valid(matrix, i, j - 1) and is_safe(matrix, visited, i, j - 1):
        min_dist = find_path(matrix, visited, i, j - 1, dest_row, dest_col, min_dist, dist + 1)

    visited[i][j] = 0

    return min_dist


def add_objects_to_map(obj_number, obj_type, N, M, j_row, j_col, t_row, t_col, map_list):
    global CHEESE_POSITION_ROW, CHEESE_POSITION_COL

    for i in range(obj_number):
        while True:
            row_position = random.randrange(N)
            col_position = random.randrange(M)
            if (row_position != j_row or col_position != j_col) and \
                    (row_position != t_row or col_position != t_col) and \
                    map_list[row_position][col_position] == 0:
                if obj_type != 'obstacle':
                    map_list[row_position][col_position] = 2
                    CHEESE_POSITION_ROW.append(row_position)
                    CHEESE_POSITION_COL.append(col_position)
                else:
                    map_list[row_position][col_position] = 1
                break


def generate_string_map(map_file_name, dynamic_map_query):
    """
    Constructs the original map in a string
    with newline between rows
    Parameters
    map_file_name: The name of the map
    Returns
    The map in a string
    """
    count = 0
    global N, M, A
    map_as_list = []
    j_row = j_col = t_row = t_col = obstacles = cheese = 0

    with open(os.path.join("maps/", map_file_name + ".txt"), "r") as map_file:
        for line in map_file.readlines():
            metadata = line.strip().split(' ')
            if count == 0:
                N = int(metadata[0])
                M = int(metadata[1])
            elif count == 1:
                map_as_list = list(map(''.join, zip(*[iter(metadata[0])] * M)))
            elif count == 2:
                A = int(metadata[0])
            elif count == 3:
                j_row = int(metadata[0])
                j_col = int(metadata[1])
                mouse_row = map_as_list[j_row]
                map_as_list[j_row] = mouse_row[:j_col] + "J" + mouse_row[j_col + 1:]
            elif count == 4:
                t_row = int(metadata[0])
                t_col = int(metadata[1])
                mouse_row = map_as_list[t_row]
                map_as_list[t_row] = mouse_row[:t_col] + "T" + mouse_row[t_col + 1:]
            count = count + 1

    state = "\n".join(map(lambda row: "".join(row), map_as_list))

    # Beautify map
    state = state.replace('1', 'X')
    state = state.replace('2', 'c')

    # state = state.replace('1', '\033[31m' + 'X' + '\033[0m')
    # state = state.replace('2', '\033[33m' + 'c' + '\033[0m')
    # state = state.replace('T', '\033[34m' + 'T' + '\033[0m')
    # state = state.replace('J', '\033[36m' + 'J' + '\033[0m')

    # print("N: %d    M: %d   A: %d" % (N, M, A))
    # print(state)
    if dynamic_map_query == 'no':
        return state, N, M, A, j_row, j_col, t_row, t_col, state.count('X'), state.count('c')
    return state


# Check if there is a path from each cheese to Jerry
# and from Tom to Jerry
def get_initial_state(map_file_name, N, M, A, j_row, j_col, t_row, t_col, obstacles, cheese):
    print('===============================================')
    print(Fore.BLUE + '\tConstructing a dynamic map' + Style.RESET_ALL)
    print('===============================================')

    global CHEESE_POSITION_ROW, CHEESE_POSITION_COL
    while True:
        map_list = [[0 for _ in range(M)] for _ in range(N)]
        results = []
        CHEESE_POSITION_ROW = []
        CHEESE_POSITION_COL = []

        # add obstacles
        add_objects_to_map(obstacles, "obstacle", N, M, j_row, j_col, t_row, t_col, map_list)

        # add cheese
        add_objects_to_map(cheese, "cheese", N, M, j_row, j_col, t_row, t_col, map_list)

        str_map = ''.join(''.join(map(str, row)) for row in map_list)

        f = open("maps/map.txt", "w")
        f.write("%d %d\n%s\n%d\n%d %d\n%d %d\n" % (N, M, str_map, A, j_row, j_col, t_row, t_col))
        f.close()

        # Generate a dynamic map
        state = generate_string_map(map_file_name, "yes")

        rows = state.split('\n')
        matrix = []
        new_row = []
        for row in rows:
            for cell in row:
                new_row.append(cell)
            matrix.append(new_row)
            new_row = []

        # print(matrix)
        visited = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

        # Check if there is a path from Tom to Jerry
        min_dist = 100000
        res = find_path(matrix, visited, j_row, j_col, t_row, t_col, min_dist, 0)
        results.append(res)

        # Check if there is a path from Jerry to each cheese
        for i in range(len(CHEESE_POSITION_ROW)):
            res = find_path(matrix, visited, j_row, j_col, CHEESE_POSITION_ROW[i], CHEESE_POSITION_COL[i], min_dist, 0)
            results.append(res)

        if min_dist in results:
            print("Tom or Jerry are blocked. Constructing a new map")
        else:
            break

    return state
