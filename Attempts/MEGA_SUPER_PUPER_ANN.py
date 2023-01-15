import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

class NeuralNetwork:
    def __init__(self, layers):
        self.countLayers = layers
        self.weights = np.random.rand()


class Layer:
    def __init__(self, neurons):
        self.countNeurons = neurons

class Neuron:
    def __init__(self, layers):
        #self.inputConnections = 
        pass
