from random import choice
import random
import math

from mainC import deserialize_state, get_position, ACTION_EFFECTS

TEMP_DISTRIBUTION = 0.6


def agent_is_blocked(state, action, cells_visited, N, M):
    str_state = deserialize_state(state)
    jerry_row, jerry_col = get_position(str_state, "J")

    next_jerry_row = jerry_row + ACTION_EFFECTS[action][0]
    next_jerry_col = jerry_col + ACTION_EFFECTS[action][1]

    visited = cells_visited[next_jerry_row][next_jerry_col]
    if visited > 1:
        # print("Jerry might be blocked... Choosing random action")
        # cells_visited[next_jerry_row][next_jerry_col] = 1
        return True, cells_visited
    return False, cells_visited


def get_least_visited_cell(state, actions, cells_visited):
    str_state = deserialize_state(state)
    jerry_row, jerry_col = get_position(str_state, "J")
    minimum_visited = None
    minimum_action = None

    for action in actions:
        next_jerry_row = jerry_row + ACTION_EFFECTS[action][0]
        next_jerry_col = jerry_col + ACTION_EFFECTS[action][1]

        visited = cells_visited[next_jerry_row][next_jerry_col]
        if minimum_visited is None or minimum_visited > visited:
            minimum_visited = visited
            minimum_action = action

    return minimum_action


class Strategy:
    """
    This class exposes 4 exploration / exploitation strategies
    """

    def __init__(self):
        pass

    def max_first(self, Q, state, legal_actions, cells_visited, N, M):
        """
        Maximum exploitation: the best action is chosen
        Parameters
        Q: a dictionary with (s,a): utility
        state: current map configuration
        legal_actions: a list with the legal actions for the given state
        Returns
        the best action to make
        """
        explored_actions = [x for x in legal_actions if (state, x) in Q]

        # Choose one of the maximum explored actions
        if explored_actions:
            max_score = max([Q[(state, x)] for x in explored_actions])
            max_actions = [x for x in explored_actions if Q[(state, x)] == max_score]

            if not max_actions:
                return choice(legal_actions), cells_visited

            rand_max_action = choice(max_actions)

            # If Jerry is stuck in a loop, choose another action
            res, cells_visited = agent_is_blocked(state, max_actions[0], cells_visited, N, M)
            if res:
                if len(explored_actions) > 1:
                    min_action = get_least_visited_cell(state, explored_actions, cells_visited)
                    return min_action, cells_visited
                # explored_actions.remove(rand_max_action)
                # return choice(explored_actions), cells_visited
                # print(legal_actions)
                return choice(legal_actions), cells_visited

            return rand_max_action, cells_visited

        # Return a random unexplored action
        rand_action = choice(legal_actions)

        # If Jerry is stuck in a loop, choose another action
        res, cells_visited = agent_is_blocked(state, rand_action, cells_visited, N, M)
        if res:
            if len(legal_actions) > 1:
                min_action = get_least_visited_cell(state, legal_actions, cells_visited)
                return min_action, cells_visited

            return choice(legal_actions), cells_visited

        return rand_action, cells_visited

    def random_action(self, state, legal_actions, cells_visited, N, M):
        """
        Returns a random action from the list
        Parameters
        legal_actions: a list with the legal actions for the given state
        Returns
        a random action
        """
        # If Jerry is stuck in a loop, make another random choice
        rand_action = choice(legal_actions)
        if len(legal_actions) > 1:
            res, cells_visited = agent_is_blocked(state, rand_action, cells_visited, N, M)
            if res:
                legal_actions.remove(rand_action)
                return choice(legal_actions), cells_visited

        # Return a random legal action
        return choice(legal_actions), cells_visited

    def exploration(self, Q, state, legal_actions, cells_visited, N, M):
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

        # Return an explored action if all actions are explored
        if not unexplored_actions:
            return choice(legal_actions), cells_visited

        rand_action = choice(unexplored_actions)

        # If Jerry is stuck in a loop, choose another unexplored action
        if len(unexplored_actions) > 1:
            res, cells_visited = agent_is_blocked(state, rand_action, cells_visited, N, M)
            if res:
                unexplored_actions.remove(rand_action)
                return choice(unexplored_actions), cells_visited

        # Return an unexplored action
        return rand_action, cells_visited

    def balanced_exploration_exploitation(self, Q, state, legal_actions):
        """
        Returns an action based on its utility.
        The higher the score, the higher the probability to get chosen.

        Parameters
        Q: the dictionary with (s,a): utility
        state: the current state
        legal_actions: a list with the legal actions for the given state

        Returns
        an action
        """
        random_number = random.uniform(0, 1)
        action_to_utility = []
        # Get all actions
        for action in legal_actions:
            if (state, action) not in Q or Q[(state, action)] <= 0.0:
                action_to_utility.append((action, 0.0))
            else:
                action_to_utility.append((action, Q[(state, action)]))

        # Normalize the values (between 0 and 1)
        sum_of_actions = sum(action_to_utility[i][1] for i in range(len(action_to_utility)))

        if sum_of_actions == 0.0:
            return choice(legal_actions)

        for i in range(len(action_to_utility)):
            action_to_utility[i] = (action_to_utility[i][0], float(action_to_utility[i][1] / sum_of_actions))

        # Choose the most appropriate action to the random number
        for i in range(len(action_to_utility)):
            if random_number <= action_to_utility[i][1]:
                rand_action = action_to_utility[i][0]
                return rand_action

        # Choose a random action
        return choice(legal_actions)
