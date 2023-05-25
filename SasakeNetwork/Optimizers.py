import numpy as np

def initilization_momentum(weights):
    v = []
    for i in range(len(weights)):
        v.append(np.zeros(weights[i].shape))
    return v

def initilization_RMS(weights):
    s = []
    for i in range(len(weights)):
        s.append(np.zeros(weights[i].shape))
    return s

# Stochastic Gradient Descent
def SGD(learning_rate, weights, valueErrors, valueLayers, iterations, inputs):
    if len(weights) > 1:
        for i in range(iterations, 0, -1):
            weights[i] -= learning_rate * np.dot((valueErrors[-(i+1)] *  valueLayers[i] * (1.0 - valueLayers[i])), np.transpose(valueLayers[i - 1]))
    weights[0] -= learning_rate * np.dot((valueErrors[-1] * valueLayers[0] * (1.0 - valueLayers[0])), np.transpose(inputs))

    return weights
# Results: 93,39% 


#SGD with Momentum
def Momentum(learning_rate, weights, valueErrors, valueLayers, iterations, inputs, v):
    beta = 0.9
    
    if len(weights) > 1:
        for i in range(iterations, 0, -1):
            v[i] = (beta * v[i]) + ((1 - beta) * np.dot((valueErrors[-(i+1)] *  valueLayers[i] * (1.0 - valueLayers[i])), np.transpose(valueLayers[i - 1])))
            weights[i] -= learning_rate * v[i]
    v[0] = (beta * v[0]) + ((1 - beta) * np.dot((valueErrors[-1] * valueLayers[0] * (1.0 - valueLayers[0])), np.transpose(inputs)))
    weights[0] -= learning_rate * v[0]

    return weights

# не работает // ???
def RMSprop(learning_rate, weights, valueErrors, valueLayers, iterations, inputs, s):
    beta = 0.9
    eps = pow(10, -4)

    if len(weights) > 1:
        for i in range(iterations, 0, -1):
            s[i] = (beta * s[i]) + ((1 - beta) * np.square(np.dot((valueErrors[-(i+1)] * valueLayers[i] * (1.0 - valueLayers[i])), np.transpose(valueLayers[i - 1]))))
            weights[i] -= learning_rate * (np.dot((valueErrors[-(i+1)] *  valueLayers[i] * (1.0 - valueLayers[i])), np.transpose(valueLayers[i - 1])) / (np.sqrt(s[i] + eps)))
    s[0] = (beta * s[0]) + ((1 - beta) * np.square(np.dot((valueErrors[-1] * valueLayers[0] * (1.0 - valueLayers[0])), np.transpose(inputs))))
    weights[0] -= learning_rate * (np.dot((valueErrors[-1] *  valueLayers[0] * (1.0 - valueLayers[0])), np.transpose(inputs)) / (np.sqrt(s[0] + eps)))

    return weights

# TODO 
# Тайминги: около 10 минут
# Оценка: ~10%


    