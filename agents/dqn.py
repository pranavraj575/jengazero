from src.agent import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
from agents.replay_buffer import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def featurize(tower: Tower, MAX_HEIGHT=INITIAL_SIZE*3):
    """
    returns handpicked features of tower as an np vector
    :param tower: input tower
    :param MAX_HEIGHT: max possible height of tower, default INITIAL_SIZE*3
    :return: np vector, (MAX_HEIGHT*2+MAX_HEIGHT*3+1)
    """
    COMs = np.zeros((MAX_HEIGHT, 2))
    blocks = np.zeros((MAX_HEIGHT, 3))

    COMs[:tower.height(), :] = [[x, y] for x, y, z in
                                tower.COMs]  # COMs[0] shoult be the overall tower COM, projected to xy
    for i, layer in enumerate(tower.block_info):
        blocks[i, :] = [t is not None for t in layer]
    return np.concatenate(
        ([tower.height()],
         COMs.flatten(),
         blocks.flatten(),
         )
    )


FEATURESIZE = INITIAL_SIZE*3*2 + INITIAL_SIZE*3*3 + 1


class DQN(nn.Module):
    def __init__(self, layers, activation=F.relu, output_activation=None):
        """
        :param layers: list of ints, dim of layers in the network
        :param activation: activation before each hidden layer
        :param output_activation: activation before output (None if no activation)
        """
        super().__init__()
        self.nn_layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        for i in range(len(layers) - 1):
            self.nn_layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.nn_layers.append(nn.Linear(layers[-1], 1))

    def forward(self, x):
        # return self.linear_relu_stack(x)
        for layer in self.nn_layers[:-1]:
            x = self.activation(layer(x))
        x = self.nn_layers[-1](x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class DQN_player(Agent):
    def __init__(self, hidden_layers, gamma=.99, lr=.001):
        super().__init__()
        layers = [FEATURESIZE*2] + hidden_layers
        self.network = DQN(layers=layers).to(device=DEVICE)
        self.target_net = DQN(layers=layers).to(DEVICE)
        self.target_net.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = ReplayMemory()
        self.info = dict()
        self.gamma = gamma

    def pick_move(self, tower: Tower, epsilon=0., network=None):
        moves = tower.valid_moves()
        if not moves:
            return None

        if np.random.random() < epsilon:
            return np.random.choice(moves)
        if network is None:
            network = self.network
        return max(moves, key=lambda action: network(self.vectorize_state_action(tower, action)))

    def vectorize_state_action(self, tower: Tower, action):
        """
        vectorizes (s,a) pair
        imagines removing a block, and placing it perfectly, and concatenates those vectors
        """
        remove, place = action
        removed = tower.remove_block(remove)
        placed = removed.place_block(place, blk_pos_std=0., blk_angle_std=0.)

        return torch.tensor(np.concatenate((featurize(removed), featurize(placed))), dtype=torch.float32,
                            device=DEVICE).reshape((1, -1))

    def save_all(self, path):
        """
        saves all info to a folder
        """
        if not os.path.exists(path):
            os.makedirs(path)

        self.buffer.save(os.path.join(path, 'buffer.pkl'))
        torch.save(self.network.state_dict(), os.path.join(path, 'model.pkl'), _use_new_zipfile_serialization=False)
        f = open(os.path.join(path, 'info.pkl'), 'wb')
        pickle.dump(self.info, f)
        f.close()

    def load_all(self, path):
        """
        loads all info from a folder
        """
        self.buffer.load(os.path.join(path, 'buffer.pkl'))
        self.network.load_state_dict(torch.load(os.path.join(path, 'model.pkl')))
        f = open(os.path.join(path, 'info.pkl'), 'rb')
        self.info = pickle.load(f)
        f.close()

    def tower_value(self, tower: Tower, network=None):
        """
        max of Q values over possible actions at Tower
        returns -1 if no possible moves (loss)

        """
        if network is None:
            network = self.network
        moves = tower.valid_moves()
        if not moves:
            return -1.
        return max(network(self.vectorize_state_action(tower, action)) for action in moves)

    def optimize_step(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return

        transitions = self.buffer.sample(batch_size)

        batch = Transition(*zip(*transitions))

        state_action_batch = torch.cat([self.vectorize_state_action(tower_from_array(state), action)
                                        for state, action in zip(batch.state, batch.action)])

        state_action_values = self.network(state_action_batch)
        reward_batch = torch.cat([torch.tensor([r], device=DEVICE) for r in batch.reward])
        non_final_mask = torch.tensor(tuple(map(lambda t: not t,
                                                batch.terminal)), device=DEVICE, dtype=torch.bool)
        next_state_values = torch.zeros(batch_size, device=DEVICE)

        with torch.no_grad():
            next_state_values[non_final_mask] = torch.cat(
                [self.tower_value(tower_from_array(arr), network=self.target_net)
                 for arr in batch.next_state if arr is not None])
        expected_state_action_values = (next_state_values*self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()


def train():
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005


if __name__ == "__main__":
    from agents.determined import FastPick
    from agents.randy import Randy
    from src.utils import *

    DIR = os.path.abspath(os.path.dirname(os.path.dirname(sys.argv[0])))
    seed(69)
    player = DQN_player([3])
    add_training_data(player.buffer, Randy(), Randy(), skip_opponent_step=True)
    add_training_data(player.buffer, Randy(), Randy(), skip_opponent_step=True)
    player.optimize_step(batch_size=3)
    quit()
    print(player.tower_value(Tower()))
    for state, action, next_state, reward, terminal in player.buffer.memory:
        print(tower_from_array(state), action, tower_from_array(next_state), reward, terminal)

    player2 = DQN_player([3])
    print(player2.tower_value(Tower()))
    test_saving = os.path.join(DIR, 'data', 'test_save')
    player.save_all(test_saving)
    player2.load_all(test_saving)
    print(player2.tower_value(Tower()))
    for state, action, next_state, reward, terminal in player2.buffer.memory:
        print(tower_from_array(state), action, tower_from_array(next_state), reward, terminal)

    for file in os.listdir(test_saving):
        os.remove(os.path.join(test_saving, file))

    quit()

    dq = DQN([2])
    optimizer = torch.optim.Adam(dq.parameters())
    x = torch.tensor([1., 2.], device=DEVICE)
    print(dq.forward(x))
    for i in range(50):
        optimizer.zero_grad()

        y = dq.forward(x)
        target = torch.tensor([.69], device=DEVICE)
        mse = nn.MSELoss()
        loss = mse(y, target)
        loss.backward()
        optimizer.step()
    print(dq.forward(x))
    torch.save(dq.state_dict(), 'test.pkl', _use_new_zipfile_serialization=False)
    dq2 = DQN([2])
    print(dq2.forward(x))
    dq2.load_state_dict(torch.load('test.pkl'))
    print(dq2.forward(x))
