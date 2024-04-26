import torch.nn as nn


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
