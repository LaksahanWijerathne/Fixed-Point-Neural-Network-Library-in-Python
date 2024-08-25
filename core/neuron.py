# neuron.py
import random
from .fixed_point_arithmetic import FixedPointArithmetic

class Neuron:
    def __init__(self, num_inputs, decimal_points):
        self.fixed_point = FixedPointArithmetic(decimal_points)
        self.weights = [random.randint(-self.fixed_point.scaling_factor, self.fixed_point.scaling_factor) for _ in range(num_inputs)]
        self.bias = random.randint(-self.fixed_point.scaling_factor, self.fixed_point.scaling_factor)

    def compute_output(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.apply_activation_function(weighted_sum)

    def apply_activation_function(self, weighted_sum):
        return max(0, weighted_sum)  # ReLU

    def train(self, inputs, target, learning_rate):
        output = self.compute_output(inputs)
        error = target - output
        self.weights = [
            w + int(learning_rate * error * x)
            for w, x in zip(self.weights, inputs)
        ]
        self.bias += int(learning_rate * error)

    def scale_input(self, input_value):
        return self.fixed_point.scale(input_value)

    def convert_output(self, output):
        return self.fixed_point.descale(output)
