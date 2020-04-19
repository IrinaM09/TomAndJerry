import numpy as np
import matplotlib.pyplot as plt


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


def plot_quality_table(Q, strategy_name):
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
    plt.title(f"Strategy: {strategy_name}\n\n" +
              f"Avg UP = {avg_up}, Avg DOWN = {avg_down}\nAvg LEFT = {avg_left}, Avg RIGHT = {avg_right}")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def plot_evaluation_batch(constant_parameter, variable_evaluators, won_games_all_strategies, train_ep,
                          constant_param_name, variable_param_name):
    columns = ('MaxFirst', 'Random', 'Exploration', 'Balanced Exploration / Exploitation')
    rows = []
    for i in variable_evaluators:
        rows.append('(%d, %d)' % (constant_parameter, i))

    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))

    data = []
    for i in range(len(won_games_all_strategies[0])):
        row = [won_games_all_strategies[0][i],
               won_games_all_strategies[1][i],
               won_games_all_strategies[2][i],
               won_games_all_strategies[3][i]]
        data.append(row)

    plt.table(cellText=data,
              rowLabels=rows,
              rowColours=colors,
              colLabels=columns,
              loc='center')
    plt.axis('off')
    plt.axis('tight')
    plt.title(f"Constant {constant_param_name}: {constant_parameter}\n" +
              f"{variable_param_name} = {variable_evaluators}. TRAIN EPISODES = {train_ep}")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
