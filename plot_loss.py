import os.path

from agents.dqn import *
from matplotlib import pyplot as plt

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 14})
    seed(69)
    DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
    epochs = 50
    correct_size = 10
    save_path = os.path.join(DIR, 'data',
                             'dqn_against_random_' + str(epochs) + '_epochs_towersize_' + str(correct_size))
    if not os.path.exists(save_path):
        quit()
    # correct_featuresize = 1 + correct_size*3*2 + correct_size*3*3
    player = DQN_player([256], max_height=correct_size*3)
    player.load_all(save_path)

    epoch_wins = player.info['test win rate']
    epochs = [epoch for epoch, _ in epoch_wins]
    wins = [win for _, win in epoch_wins]
    plt.plot(epochs, wins)
    plt.title("Win rate Against Randomly Playing Agent")

    plt.ylim((0, plt.ylim()[1]))
    plt.xlabel('Epochs')
    plt.ylabel('Proportion of games won')
    plt.show()

    possibilities = {
        (): .3,
        (0,): .2,
        (1,): .2,
        (2,): .2,
        (0, 2): .1,
    }

    if True:
        t = Tower(default_ht=correct_size)
        for L in list(range(correct_size - 2)) + [
            np.random.choice((-1, -2))]:  # one of the top level or level right below it can be unfilled
            ran = np.random.random()
            thing = None
            for thingy in possibilities:

                ran -= possibilities[thingy]
                if ran <= 0:
                    thing = thingy
                    break
            if thing == ():
                continue
            for i in thing:
                test = t.remove_block((L, i))
                if not test.falls():
                    t = test
        print(t)
        print('heatmap=', player.heatmap(tower=t))
        print('blocks={')
        for (L, layer) in enumerate(t.block_info):
            for i, t in enumerate(layer):
                if t is not None:
                    x, y, z, a = t.to_vector()
                    print((L, i), ':(', x, ',', y, ',', z, ',', a, '),')
        print('}')
