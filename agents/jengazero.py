from agents.mcts import *
from src.networks import *
from agents.replay_buffer import *
from src.tower import INITIAL_SIZE
import torch


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

        network = params['value_network']  # tower embedding -> value (of ending at a tower)
        tower_embedding = params['embedding']  # tower -> embedding
        return network(tower_embedding(self.tower))

    def policy(self, move, params):
        if self.policy_dist is None:
            network = params['policy_network']  # tower embedding -> policy
            tower_embedding = params['embedding']  # tower -> embedding
            self.policy_dist = network(tower_embedding(self.tower))
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

        output_dim = 3*max_tower_size + 3 + 1
        layers = [tower_embed_dim] + hidden_layers + [output_dim]
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant

        self.network = FFNetwork(layers)

        self.tower_embedder = tower_embedder
        self.tower_embed_dim = tower_embed_dim
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = ReplayMemory(namedtup=State_Reward_Distrib)

        self.info = dict()
        self.info['epochs_trained'] = 0
        self.info['test win rate'] = []

        def policy_network(embedding):
            """
            assumes
            """
            result = self.network(embedding)[:-1]
            pick_result = result[:-3]
            place_result = result[-3:]
            return (pick_result.reshape((-1, 1))@place_result.reshape((1, -1))).flatten()

        value_network = lambda embedding: self.network(embedding)[-1]

        def move_index_map(move):
            (L, i_remove), i_place = move
            i = L*3 + i_remove
            j = i_place
            return i*3 + j

        # params will eventually be passed to NNState.evaluate and NNState.policy
        self.params = {
            'policy_network': policy_network,
            'embedding': tower_embedder,
            'move_index_map': move_index_map,
            'value_network': value_network,
        }

    def policy_loss(self, towers, probability_dists):
        tower_embeddings = torch.tensor([self.tower_embedder(tower) for tower in towers])
        policies = self.network(tower_embeddings)[:, :-1]
        pick_result = (policies[:, :-3]).unsqueeze(2)
        # N X D X 1
        place_result = (policies[:, -3:]).unsqueeze(1)
        # N X 1 X 3
        policies = torch.bmm(pick_result, place_result).reshape((-1, 1))
        # N x |DIST|

        criterion = nn.CrossEntropyLoss()
        loss = criterion(policies, probability_dists)
        loss.backward()
        return loss

    def value_loss(self, towers, targets):
        tower_embeddings = torch.tensor([self.tower_embedder(tower) for tower in towers])
        values = self.network(tower_embeddings)[:, -1]
        criterion = nn.SmoothL1Loss()
        loss = criterion(values, targets)
        loss.backward()
        return loss

    def add_training_data(self, tower):
        best_move,root_node=mcts_search(root_state=NNState(tower=tower),
                    iterations=self.num_iterations,
                    exploration_constant=self.exploration_constant,
                    params={self.params})

    def pick_move(self, tower: Tower):
        root_state = NNState(tower=tower)
        best_move, root_node = mcts_search(root_state=root_state,
                                           iterations=self.num_iterations,
                                           exploration_constant=self.exploration_constant,
                                           params={self.params})
        return best_move
