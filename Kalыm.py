import numpy as np

I_DIM, H_DIM, O_DIM, OUT_NEURON = 2, 2, 2, 1
LEARN_RATE = 0.8
EPOCHS = 100

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

def derivative_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x) 

def mse_loss(y_true, y_pred):
    return 0.5 * (y_true - y_pred)**2

class Neuron():
    def __init__(self):
        self.InWeights = np.random.random()
        self.bias = np.random.rand()
        self.Value = 0
        self.error = 0
    
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

    def set_error(self, error):
        self.error = error

    def get_error(self):
        return self.error

    def set_InWeights(self, new_weights):
        self.InWeights = new_weights

    def Activation(self, x):
        return sigmoid(np.dot(self.InWeights, x) + self.bias)

class Network(Neuron):
    def __init__(self):
        # Data_type = Neuron()
        # self.ValueIN = np.empty(I_DIM, dtype=Data_type)
        # self.ValueHN = np.empty(H_DIM, dtype=Data_type)
        # self.ValueON = np.empty(O_DIM, dtype=Data_type)
        # self.Output = np.empty(OUT_NEURON, dtype=Data_type)
        
        self.ValueIN = np.array([Neuron() for _ in range(I_DIM)])
        self.ValueHN = np.array([Neuron() for _ in range(H_DIM)])
        self.ValueON = np.array([Neuron() for _ in range(O_DIM)])
        self.Output = np.array([Neuron() for _ in range(OUT_NEURON)])
        
        # self.ValueIN = [Neuron() for _ in range(I_DIM)]
        # self.ValueHN = [Neuron() for _ in range(H_DIM)]
        # self.ValueON = [Neuron() for _ in range(O_DIM)]
        # self.Output = [Neuron() for _ in range(OUT_NEURON)]

    def FeedForward(self, data):
        # for i in range(len(self.ValueIN)):
        #     self.ValueIN[i].inWeights(I_DIM)
        #     self.ValueIN[i].Value = sum([self.ValueIN[i].Activation(data[j]) for j in range(len(data))])
        # for i in range(len(self.ValueHN)):
        #     self.ValueHN[i].inWeights(I_DIM)
        #     self.ValueHN[i].Value = sum([self.ValueHN[i].Activation(self.ValueIN[j].Value) for j in range((np.linalg.matrix_rank(self.ValueIN[i].Value)))])
        # for i in range(len(self.ValueON)):
        #     self.ValueON[i].inWeights(H_DIM)
        #     self.ValueON[i].Value = sum([self.ValueON[i].Activation(self.ValueHN[j].Value) for j in range(np.linalg.matrix_rank(self.ValueHN[i].Value))])
        # self.Output[0].inWeights(O_DIM)
        # s = 0
        # for i in range(len(self.ValueON)):
        #     s += self.ValueON[i].Value
        #     self.Output[0].Value = self.Output[0].Activation(s)
        # return sum(self.Output[0].Value)

        for i in range(len(self.ValueIN)):
            self.ValueIN[i].inWeights(I_DIM)
            self.ValueIN[i].Value = self.ValueIN[i].Activation(data[0])
        print(self.ValueIN[0].Value)


    

    def BackProp(self, o, pred, data):
        delta_out = derivative_sigmoid(o) * (o - pred) 
        for i in range(O_DIM):
            self.ValueON[i].set_error(derivative_sigmoid(self.ValueON[i].Value) * delta_out * self.Output[0].InWeights[i])
            self.Output[0].InWeights[i] -= LEARN_RATE * delta_out * self.ValueON[i].Value
        for i in range(H_DIM):
            self.ValueHN[i].set_error(derivative_sigmoid(self.ValueHN[i].Value) * self.ValueON[i].InWeights[i] * self.ValueON[i].error)
            self.ValueON[i].InWeights[i] -= LEARN_RATE * self.ValueON[i].error * self.ValueHN[i].Value
        for i in range(I_DIM):
            self.ValueIN[i].set_error(derivative_sigmoid(self.ValueIN[i].Value) * self.ValueHN[i].InWeights[i] * self.ValueHN[i].error)
            self.ValueHN[i].InWeights[i] -= LEARN_RATE * self.ValueHN[i].error * self.ValueIN[i].Value
            self.ValueIN[i].InWeights[i] -= LEARN_RATE * self.ValueIN[i].error * data
        self.Output[0].bias -= LEARN_RATE * delta_out



    def Train(self, data, perfect_data):
        for epoch in range(EPOCHS):
            for x, y in zip(data, perfect_data):
                Out = self.FeedForward(data) 
                self.BackProp(Out, y, x)


    def Run(self, data):
        for i in range(len(self.ValueIN)):
            self.ValueIN[i].Value = self.ValueIN.Activaion(data)
        for i in range(len(self.ValueHN)):
            self.ValueHN[i].Value = self.ValueHN.Activation(self.ValueIN.Value)
        for i in range(len(self.ValueON)):
            self.ValueON[i].Value = self.ValueON.Activaion(self.ValueHN.Value)
        self.Output[0].Value = self.Output.Activation(self.ValueON.Value)
        return self.Output[0].Value
            
            


data = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])  

all_y_trues = np.array([1, 0, 1, 1])

network = Network()
network.Train(data, all_y_trues)

test_1 = np.array([0, 0])  # 0
test_2 = np.array([1, 1])  # 1
print("test_1: %.3f" % network.Run(test_1))
print("test_2: %.3f" % network.Run(test_2))
