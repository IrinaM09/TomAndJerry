import os.path
from copy import deepcopy
from random import choice, random
from time import sleep
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import clear_output

# File to read map from
MAP_NAME = "mini_map"

# Meta-parameters
LEARNING_RATE = 0.1  # @param {type: "slider", min: 0.001, max: 1.0, step: 0.01}
DISCOUNT_FACTOR = 0.99  # @param {type: "slider", min: 0.01, max: 1.0, step: 0.01}

# Probabilit to choose a random action
EPSILON = 0.05  # @param {type: "slider", min: 0.0, max:1.0, step: 0.05, default: 0.05}

# Training and evaluation episodes
TRAIN_EPISODES = 1000  # @param {type: "slider", min: 1, max: 20000, default: 1000}

# Evaluate after specified number of episodes
EVAL_EVERY = 10  # @aram {type: "slider", min: 0, max: 1000}

# Evaluate using the specified number of episodes
EVAL_EPISODES = 10  # @param {type: "slider", min: 1, max: 1000}

# Display
VERBOSE = False  # @param {type: "boolean"}
PLOT_SCORE = True  # @param {type: "boolean"}
SLEEP_TIME = 1  # @param {type: "slider", min:1, max:10}

# Show the end result
FINAL_SHOW = True  # @param {type: "boolean"}

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]

ACTION_EFFECTS = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "STAY": (0, 0)
}

MOVE_REWARD = -0.1
WIN_REWARD = 10.0
LOSE_REWARD = -10.0


# Functions to serialize / deserialize game states
def __serialize_state(state):
    return "\n".join(map(lambda row: "".join(row), state))


def __deserialize_state(str_state):
    return list(map(list, str_state.split("\n")))


def get_initial_state(map_file_name):
    """
    Constructs the original map in a string
    with newline between rows

    Parameters
    map_file_name: The name of the map

    Returns
    The map in a string
    """
    # from pathlib import Path
    # state = Path('maps/' + map_file_name + '.txt').read_text()
    count = 0
    N = 0
    M = 0
    A = 0
    map_as_list = []

    with open(os.path.join("maps/", map_file_name + ".txt"), "r") as map_file:
        for line in map_file.readlines():
            metadata = line.strip().split(' ')
            print(metadata)
            if count == 0:
                N = int(metadata[0])
                M = int(metadata[1])
            elif count == 1:
                map_as_list = list(map(''.join, zip(*[iter(metadata[0])] * M)))
                print(map_as_list)
            elif count == 2:
                A = int(metadata[0])
            elif count == 3:
                row = int(metadata[0])
                col = int(metadata[1])
                mouse_row = map_as_list[row]
                map_as_list[row] = mouse_row[:col] + 'S' + mouse_row[col + 1:]
                print("after replacement: %s" % map_as_list[row])
            elif count == 4:
                row = int(metadata[0])
                col = int(metadata[1])
                mouse_row = map_as_list[row]
                map_as_list[row] = mouse_row[:col] + 'P' + mouse_row[col + 1:]
                print("after replacement: %s" % map_as_list[row])
            count = count + 1

    print("N: %d    M: %d   A: %d" % (N, M, A))
    state = "\n".join([str(elem) for elem in map_as_list])
    print(state)

    return state


# Get the coordinates of an actor
def __get_position(state, marker):
    for row_idx, row in enumerate(state):
        if marker in row:
            return row_idx, row.index(marker)
    return -1, -1


# Check if is a final state
def is_final_state(str_state, score):
    return score < -20.0 or "S" not in str_state or "2" not in str_state


# Check if the given coordinates are valid (on map and not a wall)
def __is_valid_cell(state, row, col):
    return row >= 0 and row < len(state) and col >= 0 and col < len(state[row]) and state[row][col] != "1"


# Move to next state
def apply_action(str_state, action):
    assert (action in ACTIONS)
    message = "Soarecele moved %s." % action

    state = __deserialize_state(str_state)
    g_row, g_col = __get_position(state, "S")
    assert (g_row >= 0 and g_col >= 0)

    next_g_row = g_row + ACTION_EFFECTS[action][0]
    next_g_col = g_col + ACTION_EFFECTS[action][1]

    if not __is_valid_cell(state, next_g_row, next_g_col):
        next_g_row = g_row
        next_g_col = g_col
        message = f"{message} Not a valid cell there."

    state[g_row][g_col] = " "
    if state[next_g_row][next_g_col] == "P":
        message = f"{message} Soarecele stepped on the Pisica."
        return __serialize_state(state), LOSE_REWARD, message
    elif state[next_g_row][next_g_col] == "2":
        state[next_g_row][next_g_col] = "S"
        message = f"{message} Soarecele found 'branza'."
        return __serialize_state(state), WIN_REWARD, message
    state[next_g_row][next_g_col] = "S"

    # Pisica moves now
    b_row, b_col = __get_position(state, "P")
    assert (b_row >= 0 and b_col >= 0)

    dy, dx = next_g_row - b_row, next_g_col - b_col

    is_good = lambda dr, dc: __is_valid_cell(state, b_row + dr, b_col + dc)

    next_b_row, next_b_col = b_row, b_col
    if abs(dy) > abs(dx) and is_good(dy // abs(dy), 0):
        next_b_row = b_row + dy // abs(dy)
    elif abs(dx) > abs(dy) and is_good(0, dx // abs(dx)):
        next_b_col = b_col + dx // abs(dx)
    else:
        options = []
        if abs(dx) > 0:
            if is_good(0, dx // abs(dx)):
                options.append((b_row, b_col + dx // abs(dx)))
        else:
            if is_good(0, -1):
                options.append((b_row, b_col - 1))
            if is_good(0, 1):
                options.append((b_row, b_col + 1))
        if abs(dy) > 0:
            if is_good(dy // abs(dy), 0):
                options.append((b_row + dy // abs(dy), b_col))
        else:
            if is_good(-1, 0):
                options.append((b_row - 1, b_col))
            if is_good(1, 0):
                options.append((b_row + 1, b_col))

        if len(options) > 0:
            next_b_row, next_b_col = choice(options)

    if state[next_b_row][next_b_col] == "S":
        message = f"{message} Pisica ate Soarecele."
        reward = LOSE_REWARD
    elif state[next_b_row][next_b_col] == "2":
        message = f"{message} Pisica found branza. Soarecele lost!"
        reward = LOSE_REWARD
    else:
        message = f"{message} Pisica follows Soarecele."
        reward = MOVE_REWARD

    state[b_row][b_col] = " "
    state[next_b_row][next_b_col] = "P"

    return __serialize_state(state), reward, message


def display_state(state):
    print(state)


def get_legal_actions(str_state):
    # TODO (1) : Get the actions Greuceanu can do
    state = __deserialize_state(str_state)
    g_row, g_col = __get_position(state, "S")

    if state == 'UP' and g_row <= 0:
        return ["RIGHT", "DOWN", "LEFT", "STAY"]
    elif state == 'LEFT' and g_col <= 0:
        return ["RIGHT", "DOWN", "UP", "STAY"]
    elif state == 'DOWN' and g_row > len(state):
        return ["RIGHT", "LEFT", "UP", "STAY"]
    elif state == 'RIGHT' and g_col > len(state[g_row]):
        return ["LEFT", "DOWN", "UP", "STAY"]
    elif __is_valid_cell(state, g_row, g_col) == False:
        return [x for i, x in enumerate(ACTIONS) if i != state]
    else:
        return deepcopy(ACTIONS)


def epsilon_greedy(Q, state, legal_actions, epsilon):
    # TODO (2) : Epsilon greedy
    # special case:  explore the unexplored actions
    new_actions = []
    for action in legal_actions:
        if (state, action) not in Q:
            new_actions.append(action)

    if len(new_actions) > 0:
        return choice(new_actions)

    # exploit
    if random() > EPSILON:
        return best_action(Q, state, legal_actions)
    # explore
    else:
        return choice(legal_actions)


def best_action(Q, state, legal_actions):
    # TODO (3) : Best action
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


def q_learning():
    Q = {}
    train_scores = []
    eval_scores = []
    initial_state = get_initial_state(MAP_NAME)

    for train_ep in range(1, TRAIN_EPISODES + 1):
        clear_output(wait=True)
        score = 0
        state = deepcopy(initial_state)

        if VERBOSE:
            display_state(state);
            sleep(SLEEP_TIME)
            clear_output(wait=True)

        while not is_final_state(state, score):

            actions = get_legal_actions(state)
            action = epsilon_greedy(Q, state, actions, EPSILON)

            next_state, reward, msg = apply_action(state, action)
            score += reward

            # TODO (1) : Q-Learning
            # get the best action for next state
            max_action = best_action(Q, next_state, get_legal_actions(next_state))

            if (next_state, max_action) not in Q:
                max_Q = 0
            else:
                max_Q = Q[(next_state, max_action)]

            # the current movement might be new
            if (state, action) not in Q:
                Q[(state, action)] = 0

            # get maximum Q
            Q[(state, action)] = Q[(state, action)] + LEARNING_RATE * (
                    reward + (DISCOUNT_FACTOR * max_Q) - Q[(state, action)])

            # update state
            state = next_state

            if VERBOSE:
                print(msg);
                display_state(state);
                sleep(SLEEP_TIME)
                clear_output(wait=True)

        print(f"Episode {train_ep} / {TRAIN_EPISODES}")
        train_scores.append(score)

        # evaluate the greedy policy
        if train_ep % EVAL_EVERY == 0:
            avg_score = .0

            # TODO (4) : Evaluate
            avg_score = (avg_score + score) / EVAL_EPISODES

            eval_scores.append(avg_score)

    # --------------------------------------------------------------------------
    if FINAL_SHOW:
        state = deepcopy(initial_state)
        while not is_final_state(state, score):
            action = best_action(Q, state, get_legal_actions(state))
            state, _, msg = apply_action(state, action)
            print(msg);
            display_state(state);
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
    q_learning()
