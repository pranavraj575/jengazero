import math
from src.agent import *
from agents.randy import Randy,SmartRandy
from agents.dqn import DQN_player
from agents.mcts import MCTS_player

seed(69)
DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))

#dqn = DQN_player([256])
#dqn.load_all(os.path.join(DIR, 'data', 'dqn_against_random_50_epochs_towersize_5_hidden_layers_256_embedding_basic'))

dqn=SmartRandy()
mcts = MCTS_player()
options = [[dqn, mcts], [mcts, dqn]]
loser = [0, 0]  # dqn, mcts
for i in range(10):
    players = options[i%2]
    loser_idx, _, hist = outcome(players, Tower())
    for j, (tower, (remove, place), result) in enumerate(hist):
        if players[j%2] == dqn:
            print("dqns move ", end='')
        else:
            print('mcts move ', end='')
        print(tower, end=' ')
        print(tower.play_move_deterministic(remove, place)[0], end='')
        print()

    if players[loser_idx] == dqn:
        loser[0] += 1
        print('dqn lost')
    else:
        loser[1] += 1
        print('mcts lost')
print(loser)


def update_elos(initial_elos, num_rounds, agents):
    # initially all elos are 100
    win_counts = [[0] * len(agents)] * len(agents)
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                for _ in range(num_rounds):
                    result = outcome([agents[i], agents[j]], Tower())[0]
                    win_counts[i][j] += result
                    win_counts[j][i] += 1-result

    new_elos = initial_elos.copy()
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                # update agent i's elo based on winrate against j
                expected_score = 1.0/1.0+math.exp(-(initial_elos[i] - initial_elos[j])/400.0)
                new_elos[i] += 80 * (win_counts[i][j]/num_rounds - expected_score)
    return new_elos


