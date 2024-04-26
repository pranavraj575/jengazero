from src.agent import *
from agents.replay_buffer import *
from src.utils import *
from src.networks import *
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(FFNetwork):
    def __init__(self, layers, activation=None, output_activation=None):
        """
        LAYERS INCLUDES INPUT DIMENSION
        :param layers: list of ints, dim of layers in the network
        :param activation: activation before each hidden layer
        :param output_activation: activation before output (None if no activation)
        """
        super().__init__(layers + [1], activation=activation, output_activation=output_activation)


SKIP_OPPONENT_STEP = False


class DQN_player(Agent):
    def __init__(self,
                 hidden_layers,
                 tower_embedder,
                 tower_embed_dim,
                 max_height=None,
                 gamma=.99,
                 epsilon=.9,
                 lr=.001,
                 tau=1.
                 ):
        """
        creates an agent that uses DQN
        :param max_height: if unspecified, does the correct max_height according to INITIAL_SIZE
            only specify if loading old model
        """
        super().__init__()

        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.tower_embedder = tower_embedder
        self.tower_embed_dim = tower_embed_dim

        layers = [2*tower_embed_dim + 3] + hidden_layers
        self.network = DQN(layers=layers).to(device=DEVICE)
        self.target_net = DQN(layers=layers).to(DEVICE)
        self.update_target_net()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = ReplayMemory()

        self.info = dict()
        self.info['epochs_trained'] = 0
        self.info['test win rate'] = []

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
        L, i = remove
        new_layer = np.array(tower.boolean_blocks()[L])
        new_layer[i] = 0.
        placed = removed.place_block(place, blk_pos_std=0., blk_angle_std=0.)
        # adds in the layer after we removed a block
        return torch.tensor(np.concatenate((new_layer,
                                            self.tower_embedder(removed),
                                            self.tower_embedder(placed))), dtype=torch.float32,
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

    def load_last_checkpoint(self, path):
        """
        loads most recent checkpoint
            assumes folder name is epoch number
        """
        path = os.path.join(path, 'checkpoints')
        best = -1
        for folder in os.listdir(path):
            check = os.path.join(path, folder)
            if os.path.isdir(check) and folder.isnumeric():
                best = max(best, int(folder))
        if best < 0:
            # checkpoint not found
            return False

        self.load_all(os.path.join(path, str(best)))
        return True

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
        return max(self.q_value(tower=tower, action=action, network=network) for action in moves)

    def q_value(self, tower: Tower, action, network=None):
        """
        returns Q-value of tower-action pair
            network is self.network if not specified
        """
        if network is None:
            network = self.network

        return network(self.vectorize_state_action(tower, action))

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

        if torch.sum(non_final_mask) > 0:
            # check if there are any non-terminal states
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

    def test_against(self, agent, N=100):
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
            loser, _, hist = outcome(agents)
            if loser == idx:
                lost += 1

        self.epsilon = temp_epsilon

        # return the number of times we did not lose
        return 1 - lost/N

    def train(self, epochs=1, agent_pairs=None, testing_agent=None, checkpt_dir=None, checkpt_freq=10):
        """
        training loop
        :param epochs: total number of epochs to train
            considers epochs already trained, if non-zero
        :param agent_pairs: agent pairs to use for training data
            if None, just plays against self
        :param checkpt_dir: if specified save a model every checkpt_freq steps into a unique directory in checkpt_dir
        """
        EPS_DECAY = 1000
        if agent_pairs is None:
            agent_pairs = [(self, self)]

        if self.info['epochs_trained'] == 0 and testing_agent is not None:
            win_rate = self.test_against(testing_agent)
            print('epoch:', self.info['epochs_trained'], 'win_rate:', win_rate)
            self.info['test win rate'].append((self.info['epochs_trained'], win_rate))

        for epoch in range(epochs - self.info['epochs_trained']):
            agent1, agent2 = agent_pairs[np.random.randint(len(agent_pairs))]
            add_training_data(self.buffer, agent1=agent1, agent2=agent2, skip_opponent_step=SKIP_OPPONENT_STEP)
            loss = self.optimize_step()
            self.update_target_net()
            self.epsilon = self.epsilon*np.exp(-1/EPS_DECAY)
            self.info['epochs_trained'] += 1
            if testing_agent is not None:
                win_rate = self.test_against(testing_agent)
                print('epoch:', self.info['epochs_trained'], 'win_rate:', win_rate)
                self.info['test win rate'].append((self.info['epochs_trained'], win_rate))

            if self.info['epochs_trained']%checkpt_freq == 0:
                if checkpt_dir is not None:
                    folder = os.path.join(checkpt_dir, 'checkpoints', str(self.info['epochs_trained']))
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    self.save_all(folder)

    def heatmap(self, tower: Tower):
        """
        returns the value of each possible action removing a block in tower
            (an action consists of pick and place, we simply consider removing a block and
                take the max Q-value over all actions removing this block)
        """
        removes, places = tower.valid_moves_product()
        heat = dict()
        for remove in removes:
            for place in places:
                if remove not in heat:
                    heat[remove] = -np.inf
                    qval = self.q_value(tower=tower, action=(remove, place)).item()
                    heat[remove] = max(heat[remove], qval)
        return heat


if __name__ == "__main__":
    from agents.determined import FastPick
    from agents.randy import Randy, SmartRandy

    embedding = ('basic', basic_featurize, BASIC_FEATURESIZE)
    opponent = ('random', Randy())
    # opponent = ('smart_random', SmartRandy())

    epochs = 50
    hidden_layers = [256]
    hidden_str = ''
    for size in hidden_layers:
        hidden_str += '_' + str(size)
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    save_path = os.path.join(DIR, 'data',
                             'dqn_against_' +
                             opponent[0] + '_' +
                             str(epochs) +
                             '_epochs_towersize_' +
                             str(INITIAL_SIZE) +
                             '_hidden_layers' + hidden_str +
                             '_embedding_' + embedding[0]
                             )
    print('saving to', save_path)
    seed(42069)
    player = DQN_player(hidden_layers=hidden_layers, tower_embedder=embedding[1], tower_embed_dim=embedding[2])
    agent_pairs = [(player, player), (player, opponent[1])]
    if os.path.exists(os.path.join(save_path, 'info.pkl')):
        print('loading initial', save_path)
        player.load_all(save_path)
    elif os.path.exists(os.path.join(save_path, 'checkpoints')):
        print('loading checkpoint', save_path)
        player.load_last_checkpoint(save_path)
        print('last checkpoint:', player.info['epochs_trained'], 'epochs trained')
    else:
        print('getting games in buffer')
        player.grab_fixed_amount(128, agent_pairs=[(opponent[1], opponent[1]), ])
        print('done')

    player.train(epochs=epochs, agent_pairs=agent_pairs, testing_agent=opponent[1], checkpt_dir=save_path)

    if True:
        player.save_all(save_path)

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
