import numpy as np

I_DIM, H_DIM, O_DIM = 3, 4, 3

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

class Network():
    def __init__(self, I_DIM, H_DIM, O_DIM):
        self.ValueHN = np.empty(H_DIM, dtype=Neuron)
        self.ValueON = np.empty(O_DIM, dtype=Neuron)
        self.Weights1 = np.random.random(I_DIM * H_DIM)
        self.Weights2 = np.random.random(O_DIM * H_DIM)
        
    def FeedForward(self, data):
        for i in range(len(self.ValueHN)):
            self.ValueHN[i].inWeights(I_DIM) # переделать 
            self.ValueHN[i].Value = self.ValueHN.Activation(data)
        for i in range(len(self.ValueON)):
            self.ValueON[i].inWeights(H_DIM)
            self.ValueON[i].Value = self.ValueON.Activaion(self.ValueHN.Value)
        return sum([self.ValueON[x].Value for x in range(len(self.ValueON))])
    
    def BackProp(self):
        for i in range(len(self.ValueHN)):
            pass
        # передать полученное значение и искать ошибку
        # потом ебануть корректировку весов

    def Train(self, data, perfect_data):
        learn_rate = 0.8
        epochs = 100
        
        for epoch in range(epochs):
            # TODO че нить для данных придумать для своих
            for x, y in zip(data, perfect_data):
                Out = self.FeedForward(data)








class Neuron():
    def __init__(self):
        self.InWeights = np.random.random()
        self.bias = np.random.rand()
        self.Value = 0
    
    def get_bias(self):
        return self.bias
    
    def get_Value(self):
        return self.Value

    def set_bias(self, b):
        self.bias = b

    def set_Value(self, v):
        self.Value = v
    
    def inWeights(self, In):
        self.InWeights = np.random.random(In)

    def Activation(self, x):
        return sigmoid(np.dot(self.InWeights, x) + self.bias)
    