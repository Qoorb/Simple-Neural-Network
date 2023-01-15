import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

def mse_loss(y_true, y_pred):
    return 0.5 * (y_true - y_pred)**2

class NeuralNetwork:
    def __init__(self):
        self.w1 = 0.12
        self.w2 = 0.35
        self.w3 = -0.5
        self.w4 = 0.24
        self.w5 = -0.33
        self.w6 = 0.27
        self.w7 = -0.08
        self.w8 = 0.79
        self.w9 = 0.062
        self.w10 = 0.64

        self.b1 = 0.02
        self.b2 = -0.015
        self.b3 = -0.084
        self.b4 = 0.037
        self.b5 = 0.04
 
    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        
        h3 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        h4 = sigmoid(self.w7 * h1 + self.w8 * h2 + self.b4)

        o1 = sigmoid(self.w9 * h3 + self.w10 * h4 + self.b5)
        return o1
 
    def train(self, data, all_y_trues):
        learn_rate = 0.8
        epochs = 5000
 
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # 1 hidden layer
                sum_h1 = (self.w1 * x[0] + self.w2 * x[1] + self.b1) # numpy.float64

                h1 = sigmoid(sum_h1)

                h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
                # 2 hidden layer
                h3 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
                h4 = sigmoid(self.w7 * h1 + self.w8 * h2 + self.b4)
                # output layer
                o1 = sigmoid(self.w9 * h3 + self.w10 * h4 + self.b5)

                # расчет значений ошибок выходов нейронов
                delta_o1 = o1 * (1 - o1) * (o1 - y_true)

                delta_h4 = h4 * (1 - h4) * (delta_o1 * self.w10)
                delta_h3 = h3 * (1 - h3) * (delta_o1 * self.w9)

                delta_h2 = h2 * (1 - h2) * (delta_h4 * self.w8 + delta_h3 * self.w7)
                delta_h1 = h1 * (1 - h1) * (delta_h4 * self.w6 + delta_h3 * self.w5)

                # Коррекция весов
                # Выходной нейрон (5)
                self.w10 -= learn_rate * delta_o1 * h4
                self.w9 -= learn_rate * delta_o1 * h3
                self.b5 -= learn_rate * delta_o1
                
                # 1 нейрон 2-слоя (4)
                self.w8 -= learn_rate * delta_h4 * h1
                self.w7 -= learn_rate * delta_h4 * h2
                self.b4 -= learn_rate * delta_h4

                # 2 нейрон 2-слоя (3)
                self.w6 -= learn_rate * delta_h3 * h1
                self.w5 -= learn_rate * delta_h3 * h2
                self.b3 -= learn_rate * delta_h3

                # 1 нейрон 1-слоя (2)
                self.w4 -= learn_rate * delta_h2 * x[0]
                self.w3 -= learn_rate * delta_h2 * x[1]
                self.b2 -= learn_rate * delta_h2

                # 2 нейрон 1-слоя (1)
                self.w2 -= learn_rate * delta_h1 * x[0]
                self.w1 -= learn_rate * delta_h1 * x[1]
                self.b1 -= learn_rate * delta_h1

            if epoch % 10 == 0:
                pass
                # TODO сделать прогресс-бар
 

data = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])

all_y_trues = np.array([1, 0, 1, 1])

network = NeuralNetwork()
network.train(data, all_y_trues)

test_1 = np.array([0, 0])  # 0
test_2 = np.array([1, 1])  # 1
print("test_1: %.3f" % network.feedforward(test_1))
print("test_2: %.3f" % network.feedforward(test_2))