import random

class Neuron:
    def __init__(self, num_inputs, decimal_points):
        # Determine the scaling factor based on the desired number of decimal points
        self.scaling_factor = 10 ** decimal_points

        # Initialize weights and bias with random integers
        self.weights = [random.randint(-self.scaling_factor, self.scaling_factor) for _ in range(num_inputs)]
        self.bias = random.randint(-self.scaling_factor, self.scaling_factor)

    def compute_output(self, inputs):
        # Calculate the weighted sum of inputs
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.apply_activation_function(weighted_sum)

    def apply_activation_function(self, weighted_sum):
        # ReLU activation function: max(0, weighted_sum)
        return max(0, weighted_sum)

    def train(self, inputs, target, learning_rate):
        # Compute the current output
        output = self.compute_output(inputs)
        error = target - output

        # Update weights and bias based on the error and learning rate
        self.weights = [
            w + int(learning_rate * error * x)
            for w, x in zip(self.weights, inputs)
        ]
        self.bias += int(learning_rate * error)

    def scale_input(self, input_value):
        # Scale a floating-point input to fixed-point
        return int(input_value * self.scaling_factor)

    def convert_output(self, output):
        # Convert fixed-point output back to floating-point
        return output / self.scaling_factor

# Example usage
if __name__ == "__main__":
    neuron = Neuron(3, 2)  # Neuron with 3 inputs and precision of 2 decimal points

    # Example inputs
    inputs = [0.25, -0.75, 0.50]
    scaled_inputs = [neuron.scale_input(x) for x in inputs]

    # Compute the neuron's output
    output = neuron.compute_output(scaled_inputs)
    actual_output = neuron.convert_output(output)

    print(f"Neuron Output: {actual_output}")

    # Train the neuron with a target output (example target = 0.5)
    neuron.train(scaled_inputs, neuron.scale_input(0.5), 0.01)

    # Compute the output again after training
    output = neuron.compute_output(scaled_inputs)
    actual_output = neuron.convert_output(output)

    print(f"Neuron Output after Training: {actual_output}")
