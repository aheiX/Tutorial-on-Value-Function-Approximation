"""
This file demonstrates how to solve realizations of the Taxicab problem either to optimality or
via Value Function Approximation. It accompanies the paper "A Tutorial on Value Function Approximation for
Stochastic and Dynamic Transportation" published as an Educational Paper in the 4OR journal by Arne Heinold (https://doi.org/10.1007/s10288-023-00539-3).

Please see the GitHub-Project for updates:
https://github.com/aheiX/Tutorial-on-Value-Function-Approximation

Please send feedback to arne.heinold@bwl.uni-kiel.de.

Kiel (Germany), 24.02.2023
"""

import os
import pandas as pd
import random

# all figures will be stored in the following folder
output_folder = 'output/'

# create folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


class Realization:
    """
    This class represents a (random) realization using the parameters set in the corresponding Excel-file.
    """

    def __init__(self, input_as_series):
        # creates for all series elements an attribute of this realization
        self.__dict__ = input_as_series.to_dict()

        # set seed
        random.seed(self.rnd_seed)

        # periods
        self.periods = [t for t in range(1, self.T + 1)]
        self.last_period = self.periods[-1]

        # nodes and edges
        self.nodes = [id for id in range(1, self.V + 1)]
        self.edges = self.create_edges()

        # states
        self.states = dict()
        self.initial_state = (1, 1)
        self.add_state(node_period_tuple=self.initial_state)

        # solve realization
        if self.solution_approach == 'vfa':
            self.solve_with_vfa()
        elif self.solution_approach == 'optimal':
            self.solve_to_optimality()

        self.print_value_of_initial_state()

    def add_state(self, node_period_tuple, initial_value=0):
        """
        This functions adds a state to the realization

        :param node_period_tuple: the (node, period)-tuple identifies the state
        :param initial_value: the initial value of this (post decision) state
        :return: none
        """

        self.states[node_period_tuple] = dict(
            node=node_period_tuple[0],
            period=node_period_tuple[1],
            iteration=[0],          # list of numbers representing the iterations
            value=[initial_value]   # list of numbers representing the value of the state after the corresponding iteration
        )
        """
        Example: 
            iteration=[0, 1, 2]
            value=[0, 20, 25]
            The value of the state is estimated with 0, 20, and 25 after iterations 0, 1, and 2. 
            Note that iteration "0" represents the initial value.
        """

    def states_as_df(self):
        """
        This function creates a dataframe including all states of the realization.
        It may be used to create figures.

        :return: df as a DataFrame
        """

        data = [
            [
                self.solution_approach, self.name, self.rnd_seed,
                key, True if key == self.initial_state else False,
                state['node'], state['period'], iteration, state['value'][idx]
             ]
                for key, state in self.states.items()
                for idx, iteration in enumerate(state['iteration'])
                ]
        df = pd.DataFrame(data, columns=['solution_approach', 'name', 'rnd_seed',
                                         'state', 'is_initial_state', 'node', 'period', 'iteration', 'value'])

        return df

    def create_edges(self):
        """
           This functions creates a set of edges containing demand (d) and probability (p)
           between all nodes (i -> j) in each period (t)

           :param V: list of nodes
           :param T: list of periods
           :return: a list of edges
           """
        E = {}

        for i in self.nodes:
            for j in self.nodes:
                for t in self.periods:
                    p = random.randrange(0, 101) / 100
                    d = random.randrange(1, 101)
                    E[(i, j, t)] = dict(p=p, d=d)

        return E

    def solve_to_optimality(self):
        """
        This function computes the value of each state using backward dynamic programming,
        i.e., it applies Bellman's equation. The value is stored in self.states

        :return: n/a
        """

        # start in the last period
        for t in reversed(self.periods):
            # consider all nodes
            for source in self.nodes:
                # add state
                if (source, t) not in self.states:
                    self.add_state(node_period_tuple=(source, t))

                # set state
                current_state = self.states[(source, t)]

                # D is a list of lists containing the reward and probability
                D = []

                # loop through all possible moves (aka target nodes)
                for target in self.nodes:
                    e = self.edges[source, target, t]
                    if t == self.last_period:
                        # only immediate reward
                        D.append([e['d'], e['p']])
                    else:
                        # immediate and downstream reward
                        next_state = self.states[(target, t + 1)]
                        D.append([e['d'] + (self.y * next_state['value'][-1]), e['p']])
                        D.append([(self.y * next_state['value'][-1]), 1])

                # compute value of optimal policy (see paper for an explanation)
                D.sort(key=lambda l: l[0], reverse=True)

                rolling_probability = 0
                value_optimal_policy = 0
                for d in D:
                    value_optimal_policy += d[0] * (1 - rolling_probability) * d[1]
                    rolling_probability += (1 - rolling_probability) * d[1]
                    if rolling_probability == 1:
                        break

                current_state['iteration'] = [0, self.N]
                current_state['value'] = [value_optimal_policy, value_optimal_policy]

    def solve_with_vfa(self):
        """
        This function applies value function approximation to the realization.
        After each iteration n, the value is stored in the states.
        Note that the states therefore implicitly represent post-decision states.

        :return: n/a
        """

        for n in range(1, int(self.N) + 1):

            # initial state
            next_state = self.initial_state

            for t in self.periods:
                # select current state
                current_state = next_state

                # select source of the current state (= position of the taxi)
                source_node = self.states[current_state]['node']

                # reward_per_move stores the reward (immediate and downstream) for all possible moves
                reward_per_move = {}

                # calculate the reward for all possible moves (aka target nodes)
                for target_node in self.nodes:
                    rnd_number = random.randrange(0, 101) / 100
                    if rnd_number <= self.edges[(source_node, target_node, t)]['p']:
                        immediate_reward = self.edges[(source_node, target_node, t)]['d']
                    else:
                        immediate_reward = 0

                    post_decision_state = (target_node, t + 1)

                    if post_decision_state in self.states:
                        downstream_reward = self.states[post_decision_state]['value'][-1]
                    else:
                        downstream_reward = 0

                    reward_per_move[target_node] = immediate_reward + self.y * downstream_reward

                # select best move
                best_move = max(reward_per_move, key=reward_per_move.get)

                # set stepsize parameter
                stepsize = 0
                if self.stepsize == 'fixed':
                    stepsize = self.a
                elif self.stepsize == 'harmonic':
                    stepsize = self.a / (self.a + len(self.states[current_state]['iteration']) - 1)

                # update approximate value of the current state
                new_value = (1 - stepsize) * self.states[current_state]['value'][-1] \
                            + stepsize * reward_per_move[best_move]

                self.states[current_state]['iteration'].append(n)
                self.states[current_state]['value'].append(new_value)

                # transition
                if t < self.last_period:
                    # select greedy move (if epsilon > 0)
                    greedy_move = random.choice(self.nodes)
                    move = random.choices(population=[best_move, greedy_move],
                                          weights=[1 - self.epsilon, self.epsilon])[0]

                    # transit to next state
                    next_state = (move, t + 1)

                    # add state (if it does not exist)
                    if next_state not in self.states:
                        self.add_state(node_period_tuple=next_state)

    def print_value_of_initial_state(self):
        """
        This functions prints the value of the initial state.

        :return: n/a
        """

        print(self.name + ' (' + self.solution_approach + ')')
        print(' final value (initial state): ' + str(self.states[self.initial_state]['value'][-1]))

    def create_heatmap(self):
        """
        This function creates a heatmap with the value per state.
        The values used are optimal under the optimal policy or the latest approximation under VFA.

        :return: a figure with a heatmap
        """
        import plotly.express as px

        title = self.name + ', solution: ' + self.solution_approach

        input_matrix = []
        for v in reversed(self.nodes):
            input_matrix.append([round(self.states[(v, t)]['value'][-1], 1) if (v, t) in self.states
                                 else 0 for t in self.periods])

        fig = px.imshow(input_matrix,
                        labels=dict(x="period", y="location"),
                        x=[v for v in self.nodes],
                        y=[10 - t for t in self.periods],
                        width=700,
                        height=700,
                        text_auto=True,
                        zmin=0,
                        zmax=400,
                        color_continuous_scale='Reds',
                        origin='lower',
                        title=title
                        )
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            yaxis=dict(
                tickmode='linear',
                tick0=10,
                dtick=-1
            ),
            font_size=16
        )

        return fig

    def save_figures(self):
        """
        This function creates figures for this realization. Currently, there is only one figure created.

        :return: n/a
        """

        fig = self.create_heatmap()
        fig.write_html(output_folder + '/' + str(self) + '_heatmap.html')

    def __str__(self):
        return self.name + '_' + str(self.rnd_seed)


def figure_value_per_iteration(realizations):
    """
    This function creates a figure from all the realizations with:
        a) the iterations on the x-axis
        b) the value of the initial state on the y-axis

    The figure includes a line for each unique realization name and uses the rnd_seed as the facet_row.

    :param realizations: a list of Realization-objects
    :return: n/a
    """

    frames = [r.states_as_df() for r in realizations]
    df = pd.concat(frames)

    df = df[df['is_initial_state'].isin([True])]

    import plotly.express as px
    fig = px.line(df, x="iteration", y="value", color='name',
                  facet_row='rnd_seed',
                  title='Value of the initial state (y) after each iteration (x) for each random seed (facet row).',
                  template='simple_white')


    fig.write_html(output_folder + 'figure_value_per_iteration.html')


if __name__ == '__main__':
    """
    This function reads the Excel-file and creates a realization for each row in the sheet 'run'. 
    Please read the readme sheet in the Excel-file for further information. 
    """

    # create dataframe
    df_run = pd.read_excel('examples.xlsx', sheet_name='run')

    # create realizations for each row
    realizations = []
    for index, row in df_run.iterrows():
        new_realization = Realization(input_as_series=row)
        new_realization.save_figures()

        realizations.append(new_realization)
    
    # create a figure comparing all realizations
    figure_value_per_iteration(realizations)
