import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

def derivative_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)