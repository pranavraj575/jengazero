from agents.mcts import *
from agents.dqn import featurize, FEATURESIZE
from src.networks import FFNetwork


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


class JengaZero(Agent):
    def __init__(self):
        super().__init__()

        # params will eventually be passed to NNState.evaluate and NNState.policy
        self.params = {
            'policy_network': None,
            'embedding': None,
            'move_index_map': None,
            'value_network': None,
        }

    def pick_move(self, tower: Tower):
        root_state = NNState(tower=tower)
        mcts_search(root_state, 1000, params={self.params})
