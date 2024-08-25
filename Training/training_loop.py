# training_loop.py

def train(network, data, targets, learning_rate, epochs):
    for epoch in range(epochs):
        for inputs, target in zip(data, targets):
            network.train(inputs, target, learning_rate)
        # Optionally add code to evaluate the network's performance per epoch
