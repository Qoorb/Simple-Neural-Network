import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

def mse_loss(true, pred):
    return 0.5 * (pred - true)**2

class NeuralNetwork:
    def __init__(self, layers, neurons):
        self.countLayers = layers
        self.neurons = neurons
        self.weights = [([1] * (self.neurons[x] * self.neurons[x + 1])) for x in range(self.countLayers - 1)]
        self.bias = [1] * sum(self.neurons[1:])
    def feedForward(self, data):
        pass
    def train(self):
        pass

l = int(input("Введите количество слоев: "))
n = [int(input(f"Количество нейронов на {x + 1} слое: ")) for x in range(l)]

network = NeuralNetwork(l, n)
print(network.weights, network.neurons, sep='\n')
