import plotly.graph_objects as go


def won_games_graph(eval_batch, won_games, cheese, won_games_list, rewards_list, won_list):
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Episode', 'Games won', 'Total Reward', 'Won'],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[[i for i in range(eval_batch)],
                           [i for i in won_games_list],
                           [i for i in rewards_list],
                           [i for i in won_list]
                           ],
                   line_color='darkslategray',
                   fill_color='lightcyan',
                   align='left'))
    ])

    fig.show()
