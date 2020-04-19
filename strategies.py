from random import choice
import random
import math

from mainC import deserialize_state, get_position, ACTION_EFFECTS

TEMP_DISTRIBUTION = 0.6


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
        # print("explored actions: %s" % explored_actions)

        if explored_actions:
            max_score = max([Q[(state, x)] for x in explored_actions])
            max_actions = [x for x in explored_actions if Q[(state, x)] == max_score]

            # print("max_action: %s " % max_actions)

            if len(max_actions) == 1 and len(explored_actions) > 1:
                str_state = deserialize_state(state)
                jerry_row, jerry_col = get_position(str_state, "J")

                next_jerry_row = jerry_row + ACTION_EFFECTS[max_actions[0]][0]
                next_jerry_col = jerry_col + ACTION_EFFECTS[max_actions[0]][1]

                # if Jerry is stuck in a loop, make a random choice
                visited = cells_visited[next_jerry_row][next_jerry_col]
                # print("visited: %d" % visited)
                if visited >= max(N, M) / 2:
                    # print("Jerry might be blocked... Choosing random action")
                    cells_visited[next_jerry_row][next_jerry_col] = 1
                    return choice(explored_actions), cells_visited

            if not max_actions:
                return choice(legal_actions), cells_visited

            return choice(max_actions), cells_visited

        return choice(legal_actions), cells_visited

    def random_action(self, legal_actions):
        """
        Returns a random action from the legal list
        Parameters
        legal_actions: a list with the legal actions for the given state
        Returns
        a random action
        """
        return choice(legal_actions)

    def exploration(self, Q, state, legal_actions):
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
        for action in legal_actions:
            if (state, action) not in Q or Q[(state, action)] <= 0.0:
                action_to_utility.append((action, 0.0))
            else:
                action_to_utility.append((action, Q[(state, action)]))

        # print(action_to_utility)

        sum_of_actions = sum(action_to_utility[i][1] for i in range(len(action_to_utility)))

        if sum_of_actions == 0.0:
            # print("all states have utility 0, choosing random")
            return choice(legal_actions)

        for i in range(len(action_to_utility)):
            action_to_utility[i] = (action_to_utility[i][0], float(action_to_utility[i][1] / sum_of_actions))

        for i in range(len(action_to_utility)):
            if random_number <= action_to_utility[i][1]:
                rand_action = action_to_utility[i][0]
                # print("rand action: %s" % rand_action)
                return rand_action

        return choice(legal_actions)
