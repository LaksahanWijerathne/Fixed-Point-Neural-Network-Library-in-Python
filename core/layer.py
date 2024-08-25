# layer.py

from .neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs, decimal_points):
        self.neurons = [Neuron(num_inputs, decimal_points) for _ in range(num_neurons)]

    def forward(self, inputs):
        return [neuron.compute_output(inputs) for neuron in self.neurons]
