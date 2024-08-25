# feedforward.py

from ..core.layer import Layer

class FeedforwardNetwork:
    def __init__(self, layer_sizes, decimal_points):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i-1], decimal_points))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, inputs, targets, learning_rate):
        # Basic training logic here
        pass
