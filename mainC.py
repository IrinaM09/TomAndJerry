import os.path
from copy import deepcopy
from random import choice, random
from time import sleep
from matplotlib import pyplot as plt
import numpy as np
import math
import random
from IPython.display import clear_output
import sys
import threading
import time
import queue as Queue

# File to read map from
from map_gen import find_path

MAP_NAME = "map"

# Meta-parameters
LEARNING_RATE = 0.71  # @param {type: "slider", min: 0.001, max: 1.0, step: 0.01}
DISCOUNT_FACTOR = 0.80  # @param {type: "slider", min: 0.01, max: 1.0, step: 0.01}

# Probability to choose a random action
EPSILON = 0.05  # @param {type: "slider", min: 0.0, max:1.0, step: 0.05, default: 0.05}

# Training and evaluation episodes
TRAIN_EPISODES = 100  # @param {type: "slider", min: 1, max: 20000, default: 1000}

# Evaluate after specified number of episodes
EVAL_EVERY = 10  # @param {type: "slider", min: 0, max: 1000}

# Evaluate using the specified number of episodes
EVAL_EPISODES = 10  # @param {type: "slider", min: 1, max: 1000}

# Display
VERBOSE = False  # @param {type: "boolean"}
PLOT_SCORE = True  # @param {type: "boolean"}
SLEEP_TIME = 1  # @param {type: "slider", min:1, max:10}

# Show the end result
FINAL_SHOW = True  # @param {type: "boolean"}

# Used to put back cheese in the map
TOM_FOUND_CHEESE = False
TOM_FOUND_CHEESE_ROW = 0
TOM_FOUND_CHEESE_COL = 0

# Used to locate the cheese
CHEESE_POSITION_ROW = []
CHEESE_POSITION_COL = []

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

ACTION_EFFECTS = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "STAY": (0, 0)
}

MOVE_REWARD = -0.1
# MOVE_REWARD = 0.1
WIN_REWARD = 10.0
LOSE_REWARD = -10.0
TEMP_DISTRIBUTION = 0.6

N = 0  # number of rows
M = 0  # number of columns
A = 0  # maximum distance between Tom & Jerry


class Strategy:
    """
    This class exposes 4 exploration / exploitation strategies
    """

    def __init__(self):
        pass

    def max_first(self, Q, state, legal_actions):
        """
        Maximum exploitation: the best action is chosen
        Parameters
        Q: a dictionary with (s,a): utility
        state: current map configuration
        legal_actions: a list with the legal actions for the given state
        Returns
        the best action to make
        """
        # explored_actions = [x for x in legal_actions if (state, x) in Q]
        #
        # if explored_actions:
        #     max_score = max([Q[(state, x)] for x in explored_actions])
        #     max_actions = [x for x in explored_actions if Q[(state, x)] == max_score]
        #
        #     return choice(max_actions)
        #
        # return choice(legal_actions)
        max_action = None
        max_Q = None

        for action in legal_actions:
            # ignore unexplored actions
            if (state, action) not in Q:
                continue
            if max_Q is None or (max_Q < Q[(state, action)]):
                max_Q = Q[(state, action)]
                max_action = action

        if max_action is not None:
            return max_action
        else:
            return choice(legal_actions)

    def random_action(self, legal_actions):
        """
        Returns a random action from the legal list
        Parameters
        legal_actions: a list with the legal actions for the given state
        Returns
        a random action
        """
        return choice(legal_actions)

    def exploitation(self, Q, state, legal_actions):
        """
        Returns an unexplored action if it exists otherwise
        returns a random legal action
        Parameters
        Q: a dictionary with (s,a): utility
        state: current map configuration
        legal_actions: a list with the legal actions for the given state
        Returns
        an action
        """
        unexplored_actions = [x for x in legal_actions if (state, x) not in Q]

        return choice(unexplored_actions) if unexplored_actions else choice(legal_actions)

    def balanced_exploration_exploitation(self, Q, state, legal_actions):
        """

        Parameters
        Q: the dictionary with (s,a): utility
        state: the current state
        legal_actions: a list with the legal actions for the given state

        Returns
        """
        random_value = random.random()
        probabilities = {}
        denominator = 0.0
        if Q == {}:
            return choice(legal_actions)

        for action in legal_actions:
            if (state, action) in Q:
                print("legal action: %s" % action)
                probabilities[action] = 0
                denominator += math.exp(int(Q[(state, action)]) / TEMP_DISTRIBUTION)

        if denominator != 0.0:
            for action in legal_actions:
                if (state, action) in Q:
                    print("legal action 2: %s" % action)
                    utility = int(Q[(state, action)])
                    probabilities[action] = (math.exp(utility / TEMP_DISTRIBUTION)) / denominator

        print(probabilities)
        closest_value = list(probabilities.values())
        if not closest_value:
            return choice(legal_actions)

        # find the most appropriate probability to the random value
        res = closest_value[min(range(len(closest_value)), key=lambda i: abs(closest_value[i] - random_value))]
        return [action for (action, value) in probabilities.items() if value == res][0]


def add_objects_to_map(obj_number, obj_type, j_row, j_col, t_row, t_col, map_list):
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


# Check if there is a path from each cheese to Jerry
# and from Tom to Jerry
def get_initial_state(map_file_name, matrix_row, matrix_col, j_row, j_col, t_row, t_col, obstacles, cheese):
    global CHEESE_POSITION_ROW, CHEESE_POSITION_COL
    while True:
        map_list = [[0 for _ in range(matrix_col)] for _ in range(matrix_row)]
        results = []
        CHEESE_POSITION_ROW = []
        CHEESE_POSITION_COL = []

        # add obstacles
        add_objects_to_map(obstacles, "obstacle", j_row, j_col, t_row, t_col, map_list)

        # add cheese
        add_objects_to_map(cheese, "cheese", j_row, j_col, t_row, t_col, map_list)

        str_map = ''.join(''.join(map(str, row)) for row in map_list)

        f = open("maps/map.txt", "w")
        f.write("%d %d\n%s\n%d\n%d %d\n%d %d\n" % (N, M, str_map, A, j_row, j_col, t_row, t_col))
        f.close()

        # Generate a dynamic map
        state = generate_string_map(map_file_name)

        rows = state.split('\n')
        matrix = []
        new_row = []
        for row in rows:
            for cell in row:
                new_row.append(cell)
            matrix.append(new_row)
            new_row = []

        print(matrix)
        visited = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

        # Check if there is a path from Tom to Jerry
        min_dist = 100000
        res = find_path(matrix, visited, j_row, j_col, t_row, t_col, min_dist, 0)
        results.append(res)
        print("Path (Tom -> Jerry): %s" % res)

        # Check if there is a path from Jerry to each cheese
        for i in range(len(CHEESE_POSITION_ROW)):
            print("Check the cheese at (%d, %d): " % (CHEESE_POSITION_ROW[i], CHEESE_POSITION_COL[i]))

            res = find_path(matrix, visited, j_row, j_col, CHEESE_POSITION_ROW[i], CHEESE_POSITION_COL[i], min_dist, 0)
            results.append(res)
            print("Path (Jerry -> cheese): %s" % res)

        if min_dist in results:
            print("Tom or Jerry are blocked. Constructing a new map")
        else:
            break

    print(state + "\n--------------------------")
    return state


def generate_string_map(map_file_name):
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

    print("Constructing the map")

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
                row = int(metadata[0])
                col = int(metadata[1])
                mouse_row = map_as_list[row]
                map_as_list[row] = mouse_row[:col] + "J" + mouse_row[col + 1:]
            elif count == 4:
                row = int(metadata[0])
                col = int(metadata[1])
                mouse_row = map_as_list[row]
                map_as_list[row] = mouse_row[:col] + "T" + mouse_row[col + 1:]
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

    return state


def __serialize_state(state):
    """
    Serializes a given map configuration
    Parameters
    state: a list with the given state
    Returns
    a string with the given state
    """
    return "\n".join(map(lambda row: "".join(row), state))


def __deserialize_state(str_state):
    """
    Deserializes a given map configuration
    Parameters
    state: a string with the given state
    Returns
    a list with the given state
    """
    return list(map(list, str_state.split("\n")))


def two_points_distance(dx, dy):
    first_term = math.pow(dx, 2)
    second_term = math.pow(dy, 2)

    return math.sqrt(first_term + second_term)


# Get the coordinates of an actor
def __get_position(state, marker):
    for row_idx, row in enumerate(state):
        if marker in row:
            return row_idx, row.index(marker)
    return -1, -1


def is_final_state(str_state):
    """
    Checks if the given state is final
    (all the cheese was eaten)
    Parameters
    str_state: current state
    Returns
    True or False
    """
    return "J" not in str_state or "c" not in str_state


# Check if the given coordinates are valid (on map and not a wall)
def __is_valid_cell(state, row, col):
    return 0 <= row < len(state) and 0 <= col < len(state[row]) and state[row][col] != "X"


# Move to next state
def apply_action(str_state, action):
    global TOM_FOUND_CHEESE, TOM_FOUND_CHEESE_ROW, TOM_FOUND_CHEESE_COL
    assert (action in ACTIONS)
    message = "Jerry moves %s." % action

    # Locate Jerry
    state = __deserialize_state(str_state)
    jerry_row, jerry_col = __get_position(state, "J")
    assert (jerry_row >= 0 and jerry_col >= 0)

    # Compute next location of Jerry
    next_jerry_row = jerry_row + ACTION_EFFECTS[action][0]
    next_jerry_col = jerry_col + ACTION_EFFECTS[action][1]

    if not __is_valid_cell(state, next_jerry_row, next_jerry_col):
        next_jerry_row = jerry_row
        next_jerry_col = jerry_col
        message = f"{message} Not a valid cell there."

    state[jerry_row][jerry_col] = " "

    if state[next_jerry_row][next_jerry_col] == "T":
        message = f"{message} Jerry stepped on Tom!"
        return __serialize_state(state), LOSE_REWARD, message
    elif state[next_jerry_row][next_jerry_col] == "c":
        state[next_jerry_row][next_jerry_col] = "J"
        message = f"{message} Jerry found another cheese"
        return __serialize_state(state), WIN_REWARD, message
    state[next_jerry_row][next_jerry_col] = "J"

    # Locate Tom
    tommy_row, tommy_col = __get_position(state, "T")
    assert (tommy_row >= 0 and tommy_col >= 0)

    # Compute distance between Tom and Jerry
    dy, dx = next_jerry_row - tommy_row, next_jerry_col - tommy_col

    is_good = lambda dr, dc: __is_valid_cell(state, tommy_row + dr, tommy_col + dc)
    options = []

    # Compute next location of Tom
    next_tommy_row, next_tommy_col = tommy_row, tommy_col
    # Move up or down
    if abs(dy) > abs(dx) and is_good(dy // abs(dy), 0):
        next_tommy_row = tommy_row + dy // abs(dy)
    # Move left or right
    elif abs(dx) > abs(dy) and is_good(0, dx // abs(dx)):
        next_tommy_col = tommy_col + dx // abs(dx)
    # Move in a random direction
    else:
        if abs(dx) > 0:
            if is_good(0, dx // abs(dx)):
                options.append((tommy_row, tommy_col + dx // abs(dx)))
        else:
            if is_good(0, -1):
                options.append((tommy_row, tommy_col - 1))
            if is_good(0, 1):
                options.append((tommy_row, tommy_col + 1))
        if abs(dy) > 0:
            if is_good(dy // abs(dy), 0):
                options.append((tommy_row + dy // abs(dy), tommy_col))
        else:
            if is_good(-1, 0):
                options.append((tommy_row - 1, tommy_col))
            if is_good(1, 0):
                options.append((tommy_row + 1, tommy_col))

        if len(options) > 0:
            next_tommy_row, next_tommy_col = choice(options)

    # Check if Jerry is in danger
    if state[next_tommy_row][next_tommy_col] == "J":
        message = f"{message} Tom ate Jerry!"
        reward = LOSE_REWARD
    elif state[next_tommy_row][next_tommy_col] == "c":
        TOM_FOUND_CHEESE = True
        TOM_FOUND_CHEESE_ROW = next_tommy_row
        TOM_FOUND_CHEESE_COL = next_tommy_col
        message = f"{message} Tom found some cheese - ignore it"
        reward = LOSE_REWARD
    elif two_points_distance(dx, dy) <= float(A):
        message = f"{message} Tom is too close to Jerry. Follow him - move "
        reward = MOVE_REWARD
        actions = get_legal_actions(str_state, "T")

        min_x = None
        min_y = None
        min_action = 'UP'
        # display_state(str_state)
        # print("the distance before following is x: %d, y: %d" % (min_x, min_y))
        # print("Tom is here: (%d, %d)" % (tommy_row, tommy_col))
        # print("Jerry will be here: (%d, %d)" % (next_jerry_row, next_jerry_col))

        # Find the movement that gets the closest to Jerry
        for action in actions:
            next_tommy_row = tommy_row + ACTION_EFFECTS[action][0]
            next_tommy_col = tommy_col + ACTION_EFFECTS[action][1]
            # print("Tom will be here: (%d, %d)" % (next_tommy_row, next_tommy_col))
            next_dx = next_tommy_col - next_jerry_col
            next_dy = next_tommy_row - next_jerry_row
            # print("next tom - next jerry: (%d, %d)" % (next_dy, next_dx))
            if __is_valid_cell(state, next_tommy_row, next_tommy_col):
                if (min_x is None and min_y is None) or \
                        two_points_distance(min_x, min_y) > two_points_distance(next_dx, next_dy):
                    min_x = next_tommy_col
                    min_y = next_tommy_row
                    min_action = action
                    # print("might go %s -> x: %d y: %d" % (min_action, min_x, min_y))

        next_tommy_row, next_tommy_col = min_y, min_x
        message = f"{message + min_action}."
    else:
        message = f"{message} Tom moves random."
        reward = MOVE_REWARD

    state[tommy_row][tommy_col] = " "
    state[next_tommy_row][next_tommy_col] = "T"

    # Put the cheese back if Tom found it
    if TOM_FOUND_CHEESE and TOM_FOUND_CHEESE_ROW != next_tommy_row or TOM_FOUND_CHEESE_COL != next_tommy_col:
        state[TOM_FOUND_CHEESE_ROW][TOM_FOUND_CHEESE_COL] = "c"
        TOM_FOUND_CHEESE = False

    return __serialize_state(state), reward, message


def display_state(state):
    print(state)


def get_legal_actions(str_state, actor):
    """
    Returns a list with the valid actions for the given state
    """
    state = __deserialize_state(str_state)
    row, col = __get_position(state, actor)

    actions = [a for a in ACTIONS if __is_valid_cell(state, row + ACTION_EFFECTS[a][0], col + ACTION_EFFECTS[a][1])]

    return actions or deepcopy(ACTIONS)


def add_input(input_queue):
    while True:
        input_queue.put(sys.stdin.read(1))


def q_learning_continuous(N, M, j_row, j_col, t_row, t_col, obstacles, cheese):
    strategy = Strategy()  # one of the 4 strategies to explore the map
    Q = {}  # a dictionary with ((s,a): utility) mappings
    train_scores = []  # the scores of training
    eval_scores = []  # the scores of evaluation
    initial_state = get_initial_state(MAP_NAME, N, M, j_row, j_col, t_row, t_col, obstacles,
                                      cheese)  # the initial map configuration
    train_ep = 1
    print("You chose continuous task. Press Enter to stop the training")

    input_queue = Queue.Queue()

    input_thread = threading.Thread(target=add_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()

    last_update = time.time()

    # Train Tommy & Jerry
    while True:
        if time.time() - last_update > 0.5:
            last_update = time.time()

        if not input_queue.empty():
            train_ep = train_ep - 1
            break

        clear_output(wait=True)
        score = 0
        state = deepcopy(initial_state)

        if VERBOSE:
            display_state(state)
            sleep(SLEEP_TIME)
            clear_output(wait=True)

        # While Jerry still has cheese to eat
        while not is_final_state(state):
            # Choose a behaviour policy for Jerry
            actions = get_legal_actions(state, "J")
            # Strategy 1
            # action = strategy.max_first(Q, state, actions)
            # Strategy 2
            # action = strategy.random_action(actions)
            # Strategy 3
            # action = strategy.exploitation(Q, state, actions)
            # Strategy 4
            action = strategy.exploitation(Q, state, actions)

            next_state, reward, msg = apply_action(state, action)
            score += reward

            # Get the best action for Jerry the make next
            max_action = strategy.max_first(Q, next_state, get_legal_actions(next_state, "J"))

            # Get the utility of that action (0 if the state is new, its utility otherwise)
            max_Q = 0 if ((next_state, max_action) not in Q) else Q[(next_state, max_action)]

            # The current state might be new
            if (state, action) not in Q:
                Q[(state, action)] = 0

            # Compute the utility of current state based on the next state
            Q[(state, action)] = Q[(state, action)] + LEARNING_RATE * (
                    reward + (DISCOUNT_FACTOR * max_Q) - Q[(state, action)])

            # Update the current state
            state = next_state

            if VERBOSE:
                print(msg)
                display_state(state)
                sleep(SLEEP_TIME)
                clear_output(wait=True)

        print(f"Episode {train_ep}")

        train_scores.append(score)

        # Evaluate the policy
        if train_ep % EVAL_EVERY == 0:
            avg_score = .0

            for index in range(train_ep - EVAL_EPISODES, train_ep):
                avg_score += train_scores[index]

            avg_score /= EVAL_EPISODES
            eval_scores.append(avg_score)

        train_ep = train_ep + 1

    # --------------------------------------------------------------------------
    if FINAL_SHOW:
        state = deepcopy(initial_state)
        # Run again based on the target policy (greedy policy)
        while not is_final_state(state):
            action = strategy.max_first(Q, state, get_legal_actions(state, "J"))
            state, _, msg = apply_action(state, action)
            print(msg)
            display_state(state)
            sleep(SLEEP_TIME)
            clear_output(wait=True)

    if PLOT_SCORE:
        plt.xlabel("Episode")
        plt.ylabel("Average score")
        plt.plot(
            np.linspace(1, train_ep, train_ep),
            np.convolve(train_scores, [0.2, 0.2, 0.2, 0.2, 0.2], "same"),
            linewidth=1.0, color="blue"
        )
        plt.plot(
            np.linspace(EVAL_EVERY, train_ep, len(eval_scores)),
            eval_scores, linewidth=2.0, color="red"
        )
        plt.show()


def q_learning(N, M, j_row, j_col, t_row, t_col, obstacles, cheese):
    strategy = Strategy()  # one of the 4 strategies to explore the map
    Q = {}  # a dictionary with ((s,a): utility) mappings
    train_scores = []  # the scores of training
    eval_scores = []  # the scores of evaluation
    initial_state = get_initial_state(MAP_NAME, N, M, j_row, j_col, t_row, t_col, obstacles,
                                      cheese)  # the initial map configuration

    # Train Tommy & Jerry
    for train_ep in range(1, TRAIN_EPISODES + 1):
        print(f"Start of Episode {train_ep} / {TRAIN_EPISODES}")
        clear_output(wait=True)
        score = 0
        state = deepcopy(initial_state)

        if VERBOSE:
            display_state(state)
            sleep(SLEEP_TIME)
            clear_output(wait=True)

        # While Jerry still has cheese to eat
        while not is_final_state(state):
            # Choose a behaviour policy for Jerry
            actions = get_legal_actions(state, "J")

            # Strategy 1
            # action = strategy.max_first(Q, state, actions)

            # Strategy 2
            action = strategy.random_action(actions)

            # Strategy 3
            # action = strategy.exploitation(Q, state, actions)

            # Strategy 4
            # action = strategy.balanced_exploration_exploitation(Q, state, actions)

            next_state, reward, msg = apply_action(state, action)
            score += reward

            # Get the best action for Jerry the make next
            max_action = strategy.max_first(Q, next_state, get_legal_actions(next_state, "J"))

            # Get the utility of that action (0 if the state is new, its utility otherwise)
            max_Q = 0 if ((next_state, max_action) not in Q) else Q[(next_state, max_action)]

            # The current state might be new
            if (state, action) not in Q:
                Q[(state, action)] = 0

            # Compute the utility of current state based on the next state
            Q[(state, action)] = Q[(state, action)] + LEARNING_RATE * (
                    reward + (DISCOUNT_FACTOR * max_Q) - Q[(state, action)])

            # Update the current state
            state = next_state

            if VERBOSE:
                print(msg)
                display_state(state)
                sleep(SLEEP_TIME)
                clear_output(wait=True)

        print(f"End of Episode {train_ep} / {TRAIN_EPISODES}")
        train_scores.append(score)

        # Evaluate the policy
        if train_ep % EVAL_EVERY == 0:
            avg_score = .0

            # for index in range(train_ep - EVAL_EPISODES, train_ep):
            #     avg_score += train_scores[index]

            # avg_score /= EVAL_EPISODES
            avg_score = np.mean(train_scores[train_ep - EVAL_EVERY: train_ep])
            eval_scores.append(avg_score)

    # --------------------------------------------------------------------------
    if FINAL_SHOW:
        state = deepcopy(initial_state)
        # Run again based on the target policy (greedy policy)
        while not is_final_state(state):
            action = strategy.max_first(Q, state, get_legal_actions(state, "J"))
            state, _, msg = apply_action(state, action)
            print(msg)
            display_state(state)
            sleep(SLEEP_TIME)
            clear_output(wait=True)

    if PLOT_SCORE:
        plt.xlabel("Episode")
        plt.ylabel("Average score")
        plt.plot(
            np.linspace(1, TRAIN_EPISODES, TRAIN_EPISODES),
            np.convolve(train_scores, [0.2, 0.2, 0.2, 0.2, 0.2], "same"),
            linewidth=1.0, color="blue"
        )
        plt.plot(
            np.linspace(EVAL_EVERY, TRAIN_EPISODES, len(eval_scores)),
            eval_scores, linewidth=2.0, color="red"
        )
        plt.show()


if __name__ == '__main__':
    N = int(input("N: "))
    M = int(input("M: "))
    A = int(input("A: "))

    j_row, j_col = input("Jerry position: ").split(" ")
    j_row, j_col = int(j_row), int(j_col)
    assert (0 <= j_row < N and 0 <= j_col < M), "Jerry is outside the grid!"

    t_row, t_col = input("Tom position: ").split(" ")
    t_row, t_col = int(t_row), int(t_col)
    assert (0 <= t_row < N and 0 <= t_col < M), "Tom is outside the grid!"
    assert (t_row != j_row or t_col != j_col), "Tom can't be on the same cell as Jerry!"

    obstacles = int(input("Number of obstacles: "))
    assert (0 <= obstacles <= N * M / 2), "The number of obstacles must be between [0, N*M / 2]"

    cheese = int(input("Number of cheese: "))
    assert (1 <= cheese <= (N * M - obstacles) / 2), "The number of cheese must be between [1, (N*M - obstacles) / 2]"

    # map_list = [[0 for col in range(M)] for row in range(N)]
    #
    # # add obstacles
    # add_objects_to_map(obstacles, "obstacle", j_row, j_col, t_row, t_col, map_list)
    #
    # # add cheese
    # add_objects_to_map(cheese, "cheese", j_row, j_col, t_row, t_col, map_list)
    #
    # str_map = ''.join(''.join(map(str, row)) for row in map_list)
    #
    # f = open("maps/map.txt", "w")
    # f.write("%d %d\n%s\n%d\n%d %d\n%d %d\n" % (N, M, str_map, A, j_row, j_col, t_row, t_col))
    # f.close()

    matrix = [['X', '0', '0', '0', 'c', '0', 'X', '0', '0', '0'], ['0', '0', '0', '0', '0', 'X', '0', '0', '0', 'J'],
              ['0', '0', 'X', '0', 'X', '0', '0', '0', '0', 'X'], ['X', '0', '0', '0', '0', '0', '0', '0', '0', 'X'],
              ['0', '0', '0', 'c', '0', '0', 'X', '0', '0', 'c'], ['0', 'X', '0', 'X', '0', 'X', '0', '0', '0', '0'],
              ['0', '0', '0', 'X', 'X', '0', '0', '0', '0', 'X'], ['0', 'X', '0', 'X', '0', 'c', '0', 'c', 'X', 'X'],
              ['0', '0', '0', '0', 'X', '0', 'X', 'X', 'X', '0'], ['T', '0', '0', '0', '0', '0', 'X', 'X', '0', '0']]

    visited = [[0 for col in range(len(matrix[0]))] for row in range(len(matrix))]

    src_row = 1
    src_col = 9
    dest_row = 9
    dest_col = 0
    min_dist = find_path(matrix, visited, src_row, src_col, dest_row, dest_col, 100000, 0)

    if (min_dist != 100000):
        print("The shortest path from source to destination has length %d" % min_dist)
    else:
        print("Destination can't be reached from source")

    running_type = "STEP"

    if running_type == "STEP":
        q_learning(N, M, j_row, j_col, t_row, t_col, obstacles, cheese)
    else:
        q_learning_continuous(N, M, j_row, j_col, t_row, t_col, obstacles, cheese)
