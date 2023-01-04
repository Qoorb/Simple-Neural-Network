import numpy as np
import random 
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
    
    def set_InWeights(self, In):
        self.InWeights = np.random.random(In)

    def Activation(self, x):
        return sigmoid(np.dot(self.InWeights, x) + self.bias)
    

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))




n = Neuron()
print(n.InWeights)
n.set_InWeights(3)
print(n.InWeights)
# a = np.mat([(1,2,1), (1,2,1), (1,2,3)])
# b = np.random.rand(3,3)
# b = np.eye(3, 3)
# c = np.dot(a, b)
# f = np.empty((3, 3))
# g = np.empty(3)
# i = np.empty(3, Neuron)
#print(a, "\n", b, "\n", c, "\n", f, sep="\n")
# print(f, g, sep="\n")


# h = []
# ghf = Neuron()
# # h.append(ghf)
# # print(h[0].bias)


# a = np.empty(3, dtype=Neuron)

# a[0] = ghf
# b = random.randint(0, 100)
# print(b)

# a = np.random.rand()
# print(a)