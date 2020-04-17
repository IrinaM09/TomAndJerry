import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def won_games_graph(eval_batch, won_games, cheese, strategy_name, found_cheese_list, rewards_list, won_list):
    columns = ('Avg of Cheese found', 'Avg of Total Reward', 'Avg of Won')
    start_range = np.arange(0, eval_batch, 10).tolist()
    ep_range = []
    for i in start_range:
        ep_range.append(f'{i} - {i + 10}')

    rows = ['Episodes %s' % x for x in ep_range]

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(ep_range)))

    data = []
    for i in range(len(ep_range)):
        count_yes = won_list[i: i + 10].count('yes')
        count_no = won_list[i: i + 10].count('no')
        won = 'yes' if count_yes > count_no else 'no'

        row = [
            int(np.mean(found_cheese_list[i: i + 10])),
            np.mean(rewards_list[i: i + 10]) * 10,
            won
        ]
        data.append(row)

    plt.table(cellText=data,
              rowLabels=rows,
              rowColours=colors,
              colLabels=columns,
              loc='center')
    plt.axis('off')
    plt.axis('tight')
    plt.title(f"Strategy: {strategy_name}\n" +
              f"Games won = {won_games} / {eval_batch}. Total cheese = {cheese}")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def group_quality_table(Q):
    Q_state_moves = {}
    Q_state_scores = {}

    # Group each state by moves and by scores
    for i in range(len(Q)):
        key = list(Q)[i]
        key_state = key[0]
        key_move = key[1]
        val = list(Q.values())[i]

        if key_state not in Q_state_moves:
            Q_state_moves[key_state] = []
            Q_state_scores[key_state] = []

        Q_state_moves[key_state].append(key_move)
        Q_state_scores[key_state].append(val)

    # Get the maximum score for each state
    for i in range(len(Q_state_scores)):
        scores_key = list(Q_state_scores)[i]
        scores_val = list(Q_state_scores.values())[i]
        moves_val = list(Q_state_moves.values())[i]

        maximum = -1000000
        index = 0
        for j in range(len(scores_val)):
            if scores_val[j] >= maximum:
                maximum = scores_val[j]
                index = j

        Q_state_scores[scores_key] = maximum
        Q_state_moves[scores_key] = moves_val[index]

    return Q_state_moves, Q_state_scores


def compute_avg_moves(Q_state_moves, Q_state_scores):
    moves_dict = {}
    avg_up = avg_down = avg_left = avg_right = None

    for i in range(len(Q_state_scores)):
        key = list(Q_state_moves)[i]
        moves_val = Q_state_moves[key]
        scores_val = Q_state_scores[key]

        if moves_val not in moves_dict:
            moves_dict[moves_val] = []

        moves_dict[moves_val].append(scores_val)

    if moves_dict.get('UP'):
        avg_up = np.mean(moves_dict['UP'])
    if moves_dict.get('DOWN'):
        avg_down = np.mean(moves_dict['DOWN'])
    if moves_dict.get('LEFT'):
        avg_left = np.mean(moves_dict['LEFT'])
    if moves_dict.get('RIGHT'):
        avg_right = np.mean(moves_dict['RIGHT'])

    return avg_up, avg_down, avg_left, avg_right


def plot_quality_table_maxfirst(initial_state, Q):
    # Print the quality table
    Q_state_moves, Q_state_scores = group_quality_table(Q)

    columns = ('Max Action', 'Max Score')
    display_height = 20 if len(Q_state_moves) > 20 else len(Q_state_moves)
    rows = ['State %s' % x for x in range(display_height)]

    colors = plt.cm.BuPu(np.linspace(0, 0.5, display_height))

    data = []
    for i in range(display_height):
        key = list(Q_state_moves)[i]
        moves_val = Q_state_moves[key]
        scores_val = Q_state_scores[key]

        row = [
            moves_val,
            scores_val
        ]
        data.append(row)

    avg_up, avg_down, avg_left, avg_right = compute_avg_moves(Q_state_moves, Q_state_scores)

    plt.table(cellText=data,
              rowLabels=rows,
              rowColours=colors,
              colLabels=columns,
              loc='center')
    plt.axis('off')
    plt.axis('tight')
    plt.title(f"Strategy: MaxFirst\n\n" +
              f"Avg UP = {avg_up}, Avg DOWN = {avg_down}\nAvg LEFT = {avg_left}, Avg RIGHT = {avg_right}\n" +
              f"Map:\n" +
              f"{initial_state}")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def maxfirst_vs_random(initial_state, Q_maxfist, Q_random):
    # Print the quality table of MaxFirst strategy
    Q1_state_moves, Q1_state_scores = group_quality_table(Q_maxfist)

    columns = ('Max Action', 'Max Score')
    display_height = 20 if len(Q1_state_moves) > 20 else len(Q1_state_moves)
    rows_maxfirst = ['State %s' % x for x in range(display_height)]

    colors_maxfirst = plt.cm.BuPu(np.linspace(0, 0.5, display_height))

    data_maxfirst = []
    for i in range(display_height):
        key = list(Q1_state_moves)[i]
        moves_val = Q1_state_moves[key]
        scores_val = Q1_state_scores[key]

        row = [
            moves_val,
            scores_val
        ]
        data_maxfirst.append(row)

    avg_up, avg_down, avg_left, avg_right = compute_avg_moves(Q1_state_moves, Q1_state_scores)

    plt.table(cellText=data_maxfirst,
              rowLabels=rows_maxfirst,
              rowColours=colors_maxfirst,
              colLabels=columns,
              loc='left')
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.axis('tight')
    plt.title(f"Strategy: MaxFirst\n\n" +
              f"Avg UP = {avg_up}, Avg DOWN = {avg_down}\nAvg LEFT = {avg_left}, Avg RIGHT = {avg_right}\n" +
              f"Map:\n" +
              f"{initial_state}")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    # Print the quality table of Random strategy
    Q_state_moves, Q_state_scores = group_quality_table(Q_random)

    columns = ('Max Action', 'Max Score')
    display_height = 20 if len(Q_state_moves) > 20 else len(Q_state_moves)
    rows = ['State %s' % x for x in range(display_height)]

    colors = plt.cm.BuPu(np.linspace(0, 0.5, display_height))

    data = []
    for i in range(display_height):
        key = list(Q_state_moves)[i]
        moves_val = Q_state_moves[key]
        scores_val = Q_state_scores[key]

        row = [
            moves_val,
            scores_val
        ]
        data.append(row)

    avg_up, avg_down, avg_left, avg_right = compute_avg_moves(Q_state_moves, Q_state_scores)

    plt.table(cellText=data,
              rowLabels=rows,
              rowColours=colors,
              colLabels=columns,
              loc='right')
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.axis('tight')
    plt.title(f"Strategy: Random\n\n" +
              f"Avg UP = {avg_up}, Avg DOWN = {avg_down}\nAvg LEFT = {avg_left}, Avg RIGHT = {avg_right}\n" +
              f"Map:\n" +
              f"{initial_state}")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
