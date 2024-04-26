import torch.nn as nn
from src.tower import *


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

max_mod = 4


# give features mod 2 up to mod this number exclusive
def nim_featureize(tower: Tower):
    free_moves = tower.free_valid_moves()
    moves_till_reset = tower.blocks_on_level(-1)
    return np.array([free_moves%k for k in range(2, max_mod)] + [moves_till_reset])


NIM_FEATURESIZE = 1 + (max_mod - 2)


def union_featureize(tower: Tower, MAX_HEIGHT=INITIAL_SIZE*3):
    return np.concatenate((
        union_featureize(tower, MAX_HEIGHT=MAX_HEIGHT),
        nim_featureize(tower=tower)))


UNION_FEATURESIZE = BASIC_FEATURESIZE + NIM_FEATURESIZE
