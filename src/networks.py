import torch.nn as nn
from src.tower import *
from src.agent import Agent
import pickle


class FFNetwork(nn.Module):
    def __init__(self, layers, activation=None, output_activation=None):
        """
        LAYERS INCLUDES INPUT LAYER AND OUTPUT LAYER
        :param layers: list of ints, dim of layers in the network
        :param activation: activation before each hidden layer
        :param output_activation: activation before output (None if no activation)
        """
        super().__init__()
        self.nn_layers = nn.ModuleList()
        if activation is None:
            activation = nn.ReLU
        self.output_activation = output_activation
        for i in range(len(layers) - 2):
            self.nn_layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.nn_layers.append(activation())
        self.nn_layers.append(nn.Linear(layers[-2], layers[-1]))
        if output_activation is not None:
            self.nn_layers.append(output_activation())

    def forward(self, x):
        # return self.linear_relu_stack(x)
        for layer in self.nn_layers:
            x = layer(x)
        return x


class NetAgent(Agent):
    def __init__(self):
        super().__init__()
        self.buffer = None
        self.network = None

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

    def q_value(self, tower, action, network):
        # returns an estimate of q value of taking aciton from tower
        # does not need to be implemented, but heatmap can be visualized if it is implemented
        raise NotImplementedError

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
                    qval = self.q_value(tower=tower, action=(remove, place))
                    heat[remove] = max(heat[remove], qval)
        return heat


def basic_featurize(tower: Tower, MAX_HEIGHT=INITIAL_SIZE*3):
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


BASIC_FEATURESIZE = 1 + INITIAL_SIZE*3*2 + INITIAL_SIZE*3*3

max_mod = 4  # give features mod 2 up to mod this number exclusive


def nim_featureize(tower: Tower):
    free_moves = tower.free_valid_moves()
    layer_types = tower.layer_type_count()
    negative_moves_till_reset = tower.blocks_on_level(-1)
    full_list = [negative_moves_till_reset, free_moves] + layer_types
    return np.concatenate([np.array([item%k for item in full_list]) for k in range(2, max_mod)])


NIM_FEATURESIZE = (max_mod - 2)*(4 + 1 + 1)


def union_featureize(tower: Tower, MAX_HEIGHT=INITIAL_SIZE*3):
    return np.concatenate((
        union_featureize(tower, MAX_HEIGHT=MAX_HEIGHT),
        nim_featureize(tower=tower)))


UNION_FEATURESIZE = BASIC_FEATURESIZE + NIM_FEATURESIZE
