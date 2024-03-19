from src.agent import *
import torch.nn as nn
import torch.nn.functional as F
from agents.replay_buffer import *
from src.utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def featurize(tower: Tower, MAX_HEIGHT=INITIAL_SIZE*3):
    """
    returns handpicked features of tower as an np vector
    :param tower: input tower
    :param MAX_HEIGHT: max possible height of tower, default INITIAL_SIZE*3
    :return: np vector
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


FEATURESIZE = 1 + INITIAL_SIZE*3*2 + INITIAL_SIZE*3*3


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


SKIP_OPPONENT_STEP = True


class DQN_player(Agent):
    def __init__(self, hidden_layers, gamma=.99, epsilon=.9, lr=.001, tau=1.):
        super().__init__()

        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau

        layers = [FEATURESIZE*2+3] + hidden_layers
        self.network = DQN(layers=layers).to(device=DEVICE)
        self.target_net = DQN(layers=layers).to(DEVICE)
        self.update_target_net()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = ReplayMemory()
        self.info = dict()
        self.info['epochs_trained'] = 0

    def update_target_net(self):
        target_states = self.target_net.state_dict()
        update_states = self.network.state_dict()
        for key in update_states:
            self.target_net.state_dict()[key] = self.tau*update_states[key] + (1 - self.tau)*target_states[key]

    def pick_move(self, tower: Tower, network=None):
        moves = tower.valid_moves()
        if not moves:
            return None
        if np.random.random() < self.epsilon:
            return moves[np.random.randint(len(moves))]
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
        L,i=remove
        new_layer=np.array(tower.boolean_blocks()[L])
        new_layer[i]=0.
        placed = removed.place_block(place, blk_pos_std=0., blk_angle_std=0.)
        # adds in the layer after we removed a block
        return torch.tensor(np.concatenate((new_layer,featurize(removed), featurize(placed))), dtype=torch.float32,
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

    def optimize_step(self, batch_size=128, skip_opponent_step=SKIP_OPPONENT_STEP):
        """
        optimizes network over a batch
        :param skip_opponent_step: whether in the replay buffer, we skipped opponent step
        """

        if len(self.buffer) < batch_size:
            print('not enough training examples:', len(self.buffer))
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
                 for arr in batch.next_state if arr is not None]).flatten()
        if not skip_opponent_step:
            # in this case, this is the opponent's move
            # our value will be the negative of this value
            next_state_values = -next_state_values
        expected_state_action_values = (next_state_values*self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()
        return loss

    def grab_fixed_amount(self, n, agent_pairs=None, skip_opponent_step=SKIP_OPPONENT_STEP):
        """
        grabs a fixed amount of training examples
        :param agent_pairs: agent pairs to use for training data
            if None, just plays against self
        """
        if agent_pairs is None:
            agent_pairs = [(self, self)]

        thingy = ReplayMemory(capacity=n)
        while not thingy.full():
            agent1, agent2 = agent_pairs[np.random.randint(len(agent_pairs))]
            add_training_data(thingy, agent1=agent1, agent2=agent2, skip_opponent_step=skip_opponent_step)
        for example in thingy.memory:
            self.buffer.push(*example)

    def test_against(self, agent, N=200):
        """
        plays N games against agent, returns the success rate
            uses epsilon value of 0 (moves based on only learned Q-value)
        """
        temp_epsilon = self.epsilon
        self.epsilon = 0
        lost = 0
        for _ in range(N):
            if np.random.random() < .5:
                agents = [self, agent]
                idx = 0
            else:
                agents = [agent, self]
                idx = 1
            loser, _, _ = outcome(agents)
            if loser == idx:
                lost += 1

        self.epsilon = temp_epsilon

        # return the number of times we did not lose
        return 1 - lost/N

    def train(self, epochs=1, agent_pairs=None, testing_agent=None):
        """
        training loop
        :param agent_pairs: agent pairs to use for training data
            if None, just plays against self
        """
        EPS_DECAY = 1000
        if agent_pairs is None:
            agent_pairs = [(self, self)]

        for epoch in range(epochs):
            agent1, agent2 = agent_pairs[np.random.randint(len(agent_pairs))]
            add_training_data(self.buffer, agent1=agent1, agent2=agent2, skip_opponent_step=SKIP_OPPONENT_STEP)
            loss = self.optimize_step()
            self.update_target_net()
            self.epsilon = self.epsilon*np.exp(-1/EPS_DECAY)
            self.info['epochs_trained'] += 1
            if testing_agent is not None:
                print(self.test_against(testing_agent))


if __name__ == "__main__":
    from agents.determined import FastPick
    from agents.randy import Randy

    DIR = os.path.abspath(os.path.dirname(os.path.dirname(sys.argv[0])))
    seed(69)
    player = DQN_player([256])
    agent_pairs = [(player, player), (player, Randy())]
    player.grab_fixed_amount(128, agent_pairs=agent_pairs)

    print('win rate initial', player.test_against(Randy()))

    player.train(epochs=100, agent_pairs=agent_pairs, testing_agent=Randy())

    print('win rate final', player.test_against(Randy()))
    # add_training_data(player.buffer, Randy(), Randy(), skip_opponent_step=SKIP_OPPONENT_STEP)
    # add_training_data(player.buffer, Randy(), Randy(), skip_opponent_step=SKIP_OPPONENT_STEP)
    # player.optimize_step(batch_size=3)
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