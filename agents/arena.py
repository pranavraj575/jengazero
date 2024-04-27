import math
from src.agent import *
from agents.randy import Randy, SmartRandy
from agents.dqn import DQN_player
from agents.mcts import MCTS_player
from agents.jengazero import JengaZero
from src.networks import *

seed(69)
DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))


def update_elos(agents, num_rounds=1, initial_elos=None):
    if initial_elos is None:
        initial_elos = [100 for _ in range(len(agents))]
    # initially all elos are 100
    win_counts = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                for _ in range(num_rounds):
                    print('agent', i, 'vs agent', j, end=': ')
                    loser_index, _, hist = outcome([agents[i], agents[j]], Tower())
                    win_counts[i][j] += loser_index  # loser_index 1 means i won against j
                    win_counts[j][i] += 1 - loser_index  # loser_index 0 means j won against i
                    print('agent', (str(i) + str(j))[1 - loser_index], 'won')
                    for k, (tower, (remove, place), result) in enumerate(hist):
                        print('agent', (str(i) + str(j))[k%2], 'moved:', end=' ')
                        print(tower, end=' -> ')
                        print(tower.play_move_deterministic(remove, place)[0], end='')
                        print()
                    print()
    for row in win_counts:
        print('\t'.join([str(t) for t in row]))
    new_elos = initial_elos.copy()
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                # update agent i's elo based on winrate against j
                expected_score = 1.0/(1.0 + math.pow(10, -(initial_elos[i] - initial_elos[j])/400.0))
                real_score = win_counts[i][j]/(2*num_rounds)
                new_elos[i] += 80*(real_score - expected_score)
    return new_elos


if __name__ == '__main__':
    seed(69)
    num_iterations = 100  # for mcts
    dqn_basic = DQN_player([256], basic_featurize, BASIC_FEATURESIZE)
    dqn_basic.load_all(
        os.path.join(DIR, 'data', 'dqn_against_random_50_epochs_towersize_5_hidden_layers_256_embedding_basic'))
    dqn_nim = DQN_player([256], nim_featureize, NIM_FEATURESIZE)
    # dqn_nim.load_all(os.path.join(DIR, 'data', 'dqn_against_random_50_epochs_hidden_layers_256_embedding_nim'))

    jz_nim = JengaZero([128, 128], nim_featureize, NIM_FEATURESIZE, num_iterations=num_iterations)
    jz_nim.load_all(os.path.join(DIR, 'jengazero_data', 'jengazero_nim_featureset'))
    jz_union = JengaZero([128, 128], union_featureize, UNION_FEATURESIZE, num_iterations=num_iterations)
    jz_union.load_all(os.path.join(DIR, 'jengazero_data', 'jengazero_union_featureset'))

    all_agents = [
        Randy(),
        SmartRandy(),
        MCTS_player(num_iterations=num_iterations),
        dqn_basic,
        # dqn_nim,
        jz_nim,
        # jz_union
    ]
    new_elos = update_elos(all_agents)
    print(new_elos)
