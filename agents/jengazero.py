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
        output_dim=3*max_tower_size+1
        layers=[tower_embed_dim]+hidden_layers+[output_dim]
        self.num_iterations=num_iterations
        self.exploration_constant=exploration_constant

        self.network=FFNetwork(layers)
        self.target_network=FFNetwork(layers)

        self.tower_embed_dim=tower_embed_dim
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = ReplayMemory()

        self.info = dict()
        self.info['epochs_trained'] = 0
        self.info['test win rate'] = []

        # params will eventually be passed to NNState.evaluate and NNState.policy
        self.params = {
            'policy_network': None,
            'embedding': tower_embedder,
            'move_index_map': None,
            'value_network': None,
        }


    def pick_move(self, tower: Tower):
        root_state = NNState(tower=tower)
        best_move, root_node=mcts_search(root_state=root_state,
                                         iterations=self.num_iterations,
                                         exploration_constant=self.exploration_constant,
                                         params={self.params})
        return best_move