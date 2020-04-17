import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def won_games_graph(eval_batch, won_games, cheese, found_cheese_list, rewards_list, won_list):
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
    plt.title(f"Games won = {won_games} / {eval_batch} . Total cheese = {cheese}")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
