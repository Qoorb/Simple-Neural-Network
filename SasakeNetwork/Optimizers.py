import numpy as np

# Stochastic Gradient Descent
def SGD(learning_rate, weights, valueErrors, valueLayers, iterations, inputs):
    if len(weights) > 1:
        for i in range(iterations, 0, -1):
            weights[i] -= learning_rate * np.dot((valueErrors[-(i+1)] *  valueLayers[i] * (1.0 - valueLayers[i])), np.transpose(valueLayers[i - 1]))
    weights[0] -= learning_rate * np.dot((valueErrors[-1] * valueLayers[0] * (1.0 - valueLayers[0])), np.transpose(inputs))

    return weights


def Momentum():
    pass

#TODO other optimizers...