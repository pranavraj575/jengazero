import os.path

from agents.mcts import *
from src.networks import *
from agents.replay_buffer import *
from src.tower import INITIAL_SIZE
import torch
import time
import scipy


class NNState(State):
    def __init__(self,
                 tower=None,
                 parent=None,
                 last_move=None,
                 log_stable_prob=0.0,
                 ):
        super().__init__(tower=tower,
                         parent=parent,
                         last_move=last_move,
                         log_stable_prob=log_stable_prob
                         )
        self.policy_dist = None

    def make_move(self, move):
        new_tower, log_stable_prob = self.tower.play_move_log_probabilistic(move[0], move[1])
        return NNState(tower=new_tower,
                       parent=self,
                       last_move=move,
                       log_stable_prob=log_stable_prob)

    def evaluate(self, fell, params):
        easy_result = super().evaluate(fell=fell, params=params)
        if easy_result is not None:
            return easy_result

        network = params['value_network_from_tower']  # tower -> value (of ending at a tower)
        return network(self.tower)

    def policy(self, move, params):
        if self.policy_dist is None:
            network = params['policy_network_from_tower']  # tower -> policy
            self.policy_dist = network(self.tower)
        move_index_map = params['move_index_map']  # move -> index (of policy network output)
        return self.policy_dist[move_index_map(move)]


class JengaZero(NetAgent):
    def __init__(self,
                 hidden_layers,
                 tower_embedder,
                 tower_embed_dim,
                 max_tower_size=3*INITIAL_SIZE,
                 num_iterations=1000,
                 exploration_constant=2*math.sqrt(2),
                 lr=.001
                 ):
        super().__init__()

        self.policy_output_size = 3*max_tower_size + 3
        output_dim = self.policy_output_size + 1
        layers = [tower_embed_dim] + hidden_layers + [output_dim]
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant

        self.network = FFNetwork(layers)

        self.tower_embedder = tower_embedder
        self.tower_embed_dim = tower_embed_dim
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = ReplayMemory(capacity=1024, namedtup=State_Reward_Distrib)

        self.info = dict()
        self.info['epochs_trained'] = 0
        self.info['test win rate'] = []
        self.info['losses'] = []

        def value_network(embedding):
            return self.network(embedding)[-1]

        # params will eventually be passed to NNState.evaluate and NNState.policy
        self.params = {
            'policy_network_from_tower': lambda tower: self.policy_network_from_towers([tower]).flatten(),
            'move_index_map': self.move_index_map,
            'value_network_from_tower': lambda tower: self.value_network_from_towers([tower]).flatten(),
        }

    # def load_all(self,path):
    #     super().load_all(path)
    #     self.buffer=transfer_SRD(self.buffer)
    #     self.save_all(path)

    def policy_network_from_towers(self, towers):
        batch_size = len(towers)
        valid_moves = [tower.valid_moves_product() for tower in towers]
        mask = torch.ones((batch_size, self.policy_output_size), dtype=torch.bool)
        for k, (removes, places) in enumerate(valid_moves):
            for (L, i) in removes:
                mask[k, 3*L + i] = False
            for i in places:
                mask[k, i - 3] = False  # either -3,-2,-1, so still works

        embeddings = torch.stack([torch.tensor(self.tower_embedder(tower), dtype=torch.float) for tower in towers],
                                 dim=0)

        return self.policy_network(embeddings=embeddings, mask=mask)

    def policy_network(self, embeddings, mask):
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
        pre_softmax_condensed = self.network(embeddings)[:, :-1]
        pre_softmax_condensed = pre_softmax_condensed.masked_fill(mask, float(-1e32))
        condensed = torch.zeros_like(pre_softmax_condensed)
        condensed[:, :-3] = torch.nn.Softmax(dim=-1)(pre_softmax_condensed[:, :-3])
        condensed[:, -3:] = torch.nn.Softmax(dim=-1)(pre_softmax_condensed[:, -3:])

        return self.large_policy_from_condensed(condensed)

    def eval_of_STARTING_at_tower(self, tower: Tower):
        """
        returns value of STARTING at tower
        note this is different from value network, which returns value of ENDING at tower
        takes max of q-values
        """
        if tower.has_valid_moves():
            return max(self.q_value(tower, action=action) for action in tower.valid_moves())
        else:
            return -1

    def q_value(self, tower, action, network=None):
        return self.q_value_from_policy_net(tower=tower, action=action)
        return self.q_value_from_value_net(tower=tower, action=action)

    def q_value_from_policy_net(self, tower: Tower, action):
        policy = self.policy_network_from_towers([tower]).flatten()
        return policy[self.move_index_map(action)].item()

    def q_value_from_value_net(self, tower: Tower, action):
        """
        returns Q-value of tower-action pair
            network is self.network if not specified
        """
        remove, place = action
        guess_tower, log_prob_stable = tower.play_move_log_probabilistic(remove, place)
        prob_stable = math.exp(log_prob_stable)
        if guess_tower.has_valid_moves():
            guesstimate = self.value_network_from_towers([guess_tower]).flatten().item()
        else:
            # here we win if the tower does not fall
            guesstimate = 1

        # if falls then q-value is -1, otherwise, guess it with tower
        return prob_stable*guesstimate + (1 - prob_stable)*(-1)

    def value_network_from_towers(self, towers):
        embeddings = torch.stack([torch.tensor(self.tower_embedder(tower), dtype=torch.float) for tower in towers],
                                 dim=0)
        return self.value_network(embeddings)

    def value_network(self, embeddings):
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
        return self.network(embeddings)[:, -1]

    def move_index_map(self, move):
        (L, i_remove), i_place = move
        i = L*3 + i_remove
        j = i_place
        return i*3 + j

    def large_policy_from_condensed(self, probability_dist):
        batch_size, _ = probability_dist.shape
        # N X D X 1
        pick_result = (probability_dist[:, :-3]).unsqueeze(2)

        # N X 1 X 3
        place_result = (probability_dist[:, -3:]).unsqueeze(1)

        # N x |DIST|
        return torch.bmm(pick_result, place_result).reshape((batch_size, -1))

    def policy_loss(self, towers, probability_dist_targets, policy_smoothing=0.):
        # N x |DIST|
        large_targets = self.large_policy_from_condensed(probability_dist_targets)

        policies = self.policy_network_from_towers(towers)

        # crossentropy loss
        # annoying to use torch.nn.CrossEntropyLoss since it assumes the output is pre softmax
        # so, just implement it manually like this
        policy_losses = -torch.sum(large_targets*torch.log(policy_smoothing + policies), dim=1)

        # calculate the entropy of the target distribution, and subtract it
        # all this does is shift the loss so that the optimal loss is 0
        # this will not affect the gradients at all, as this is a constant wrt the network params
        optimal_policy_losses = -torch.sum(large_targets*torch.log(large_targets), dim=1)

        overall_loss = policy_losses - optimal_policy_losses

        return overall_loss.mean()  # average across N

    def value_loss(self, towers, targets):
        tower_embeddings = torch.stack([self.tower_embedder(tower) for tower in towers], dim=0)
        values = self.network(tower_embeddings)[:, -1]
        criterion = nn.SmoothL1Loss()
        loss = criterion(values, targets)
        return loss

    def add_training_data(self, tower, depth=float('inf')):
        if depth <= 0:
            return
        # print('adding data for ', end='')
        # print(tower)
        state = NNState(tower=tower)
        best_move, root_node = mcts_search(root_state=state,
                                           iterations=self.num_iterations,
                                           exploration_constant=self.exploration_constant,
                                           params=self.params,
                                           mode='alphazero')
        # UNORDERED q_value vector of moves to values
        q_value_vector = [child.get_exploit_score() for child in root_node.children]
        moves_taken = [child.state.last_move for child in root_node.children]

        broken_distribution = torch.zeros(self.policy_output_size)
        unordered_distribution = torch.nn.Softmax(dim=-1)(torch.tensor(q_value_vector))
        pick_distribution = torch.zeros(self.policy_output_size - 3)
        place_distribution = torch.zeros(3)
        for i in range(len(unordered_distribution)):
            (L, pick_i), place = moves_taken[i]
            place_distribution[place] += unordered_distribution[i]
            pick_distribution[3*L + pick_i] += unordered_distribution[i]

        broken_distribution[:-3] = pick_distribution
        broken_distribution[-3:] = place_distribution

        # note that the value target is negated.
        # the evaluation measures the value of ENDING at a state
        # the max of the q values is the value of STARTING at a state
        self.buffer.push(tower.to_array(), -max(q_value_vector), broken_distribution)

        next_state = state.make_move(best_move)
        if random.random() > math.exp(next_state.log_stable_prob):
            return
        if next_state.num_legal_moves == 0:
            return
        self.add_training_data(next_state.tower, depth=depth - 1)

    def training_step(self, batch_size):
        sample = self.buffer.sample(batch_size)

        batch = State_Reward_Distrib(*zip(*sample))
        towers = [tower_from_array(arr) for arr in batch.state]
        rewards = torch.tensor(batch.reward)
        distributions = torch.stack(batch.distribution, dim=0)
        self.optimizer.zero_grad()
        val_loss = self.value_loss(towers, rewards)
        pol_loss = self.policy_loss(towers, distributions)
        overall_loss = val_loss + pol_loss
        overall_loss.backward()
        self.optimizer.step()
        return val_loss, pol_loss

    def train(self, epochs=1, testing_agent=None, checkpt_dir=None, checkpt_freq=10, batch_size=128):
        testing_N = 10
        if self.info['epochs_trained'] == 0 and testing_agent is not None:
            win_rate = self.test_against(testing_agent, N=testing_N)
            print('epoch:', self.info['epochs_trained'], 'win_rate:', win_rate)
            self.info['test win rate'].append((self.info['epochs_trained'], win_rate))

        while self.buffer.size() < batch_size:
            print('adding training data to get', self.buffer.size(), 'to batch size', batch_size)
            self.add_training_data(Tower(), depth=batch_size - self.buffer.size())
        already_epoched = self.info['epochs_trained']
        for epoch in range(epochs - already_epoched):
            starting_time = time.time()
            print('training epoch ', self.info['epochs_trained'])
            self.add_training_data(Tower())
            val_loss, pol_loss = self.training_step(batch_size=batch_size)
            self.info['epochs_trained'] += 1
            self.info['losses'].append((self.info['epochs_trained'], val_loss.item(), pol_loss.item()))
            print('losses', self.info['losses'][-1])
            print('time', round(time.time() - starting_time), 'seconds')
            print()

            if testing_agent is not None:
                win_rate = self.test_against(testing_agent, N=testing_N)
                print('epoch:', self.info['epochs_trained'], 'win_rate:', win_rate)
                self.info['test win rate'].append((self.info['epochs_trained'], win_rate))

            if self.info['epochs_trained']%checkpt_freq == 0:
                if checkpt_dir is not None:
                    folder = os.path.join(checkpt_dir, 'checkpoints', str(self.info['epochs_trained']))
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    self.save_all(folder)
                    self.save_all(checkpt_dir)  # save to the folder itself as well

    def pick_move(self, tower: Tower, num_iterations=None):
        if num_iterations is None:
            num_iterations = self.num_iterations
        root_state = NNState(tower=tower)
        best_move, root_node = mcts_search(root_state=root_state,
                                           iterations=num_iterations,
                                           exploration_constant=self.exploration_constant,
                                           params=self.params,
                                           mode="alphazero")
        return best_move


if __name__ == '__main__':
    seed(69)

    DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))

    features = ('nim', nim_featureize, NIM_FEATURESIZE)
    # features = ('union', union_featureize, UNION_FEATURESIZE)

    ident = 'jengazero_' + features[0] + '_featureset'

    save_path = os.path.join(DIR, 'jengazero_data', ident)

    agent = JengaZero([128, 128],
                      num_iterations=1000,
                      tower_embedder=lambda tower:
                      torch.tensor(features[1](tower), dtype=torch.float),
                      tower_embed_dim=features[2])
    if agent.loadable(save_path):
        print('loading from', save_path)
        agent.load_all(save_path)
    else:
        checked = agent.load_last_checkpoint(save_path)
        if checked:
            print('loading from', save_path)
        else:
            print('saving to', save_path)
    agent.train(epochs=420, checkpt_freq=10, checkpt_dir=save_path)
    t = Tower()
    # t, _ = t.play_move_log_probabilistic((0, 2), 1)
    print(t)
    print(agent.heatmap(t))
