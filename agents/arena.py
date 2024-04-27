import math
import os.path

from src.agent import *
from agents.randy import Randy, SmartRandy
from agents.dqn import DQN_player
from agents.mcts import MCTS_player
from agents.jengazero import JengaZero
from src.networks import *
import pickle


def update_elos(agents, num_rounds=1, initial_elos=None, initial_win_counts=None):
    if initial_elos is None:
        initial_elos = [1000 for _ in range(len(agents))]
    # initially all elos are 100
    if initial_win_counts is None:
        initial_win_counts = np.array([[0 for _ in range(len(agents))] for _ in range(len(agents))])
    # adds 2*num_rounds to the total number of games played
    total_games=2*num_rounds+initial_win_counts[0][1]+initial_win_counts[1][0]
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                for _ in range(num_rounds):
                    print('agent', i, 'vs agent', j, end=': ')

                    loser_index, _, hist = outcome([agents[i], agents[j]], Tower())
                    initial_win_counts[i][j] += loser_index  # loser_index 1 means i won against j
                    initial_win_counts[j][i] += 1 - loser_index  # loser_index 0 means j won against i

                    print('agent', (str(i) + str(j))[1 - loser_index], 'won')

    new_elos = initial_elos.copy()
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                # update agent i's elo based on winrate against j
                expected_score = 1.0/(1.0 + math.pow(10, -(initial_elos[i] - initial_elos[j])/400.0))
                real_score = initial_win_counts[i][j]/total_games
                new_elos[i] += 69*(real_score - expected_score)
    return new_elos, initial_win_counts


def save_all(path, elos, win_counts):
    elo_file = os.path.join(path, 'data.pkl')
    f = open(elo_file, 'wb')
    pickle.dump((elos, win_counts), f)
    f.close()


def load_all(path):
    f = open(os.path.join(path, 'data.pkl'), 'rb')
    return pickle.load(f)


def loadable(path):
    return os.path.exists(os.path.join(path, 'data.pkl'))


if __name__ == '__main__':
    seed(69)
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))

    save_dir = os.path.join(DIR, 'games', 'all_agents')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_iterations = 420  # for mcts
    dqn_basic = DQN_player([256], basic_featurize, BASIC_FEATURESIZE)
    dqn_basic.load_all(
        os.path.join(DIR, 'data', 'dqn_against_random_50_epochs_towersize_5_hidden_layers_256_embedding_basic'))
    dqn_nim = DQN_player([256], nim_featureize, NIM_FEATURESIZE)
    dqn_nim.load_all(os.path.join(DIR, 'data', 'dqn_against_random_50_epochs_hidden_layers_256_embedding_nim'))

    jz_nim = JengaZero([128, 128], nim_featureize, NIM_FEATURESIZE, num_iterations=num_iterations)
    jz_nim.load_all(os.path.join(DIR, 'jengazero_data', 'jengazero_nim_featureset'))
    jz_union = JengaZero([128, 128], union_featureize, UNION_FEATURESIZE, num_iterations=num_iterations)
    jz_union.load_all(os.path.join(DIR, 'jengazero_data', 'jengazero_union_featureset'))

    for _ in range(1000):
        all_agents = [
            Randy(),
            SmartRandy(),
            dqn_nim,
            dqn_basic,
            MCTS_player(num_iterations=num_iterations),
            jz_nim,
            jz_union
        ]
        if loadable(save_dir):
            old_elos, old_win_counts = load_all(save_dir)
        else:
            old_elos = [1000 for _ in range(len(all_agents))]
            old_win_counts = np.array([[0 for _ in range(len(all_agents))] for _ in range(len(all_agents))])

        new_elos, new_win_counts = update_elos(all_agents, initial_elos=old_elos, initial_win_counts=old_win_counts)
        # new_win_counts = old_win_counts + win_counts
        save_all(save_dir, new_elos, new_win_counts)
        print(new_win_counts)
        print(new_elos)
