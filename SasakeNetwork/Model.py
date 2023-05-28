import numpy as np
import Optimizers
import Layer
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

def derivative_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)

class FeedForwardNeuralNetwork:
    
    igrek = []
    weights = []
    valueLayers = []
    valueErrors = []


    # Инициализация ИНС
    def __init__(self, inputnodes, outputnodes, learningrate, optimizer):
        # Входные, скрытые и выходные нейроны
        self.inodes = inputnodes 
        self.hnodes = []
        self.onodes = outputnodes
        self.optFunc = optimizer
        if self.optFunc == 'Momentum':
            self.v = Optimizers.initilization_momentum(FeedForwardNeuralNetwork.weights)
        elif self.optFunc == 'RMSprop':
            self.s = Optimizers.initilization_RMS(FeedForwardNeuralNetwork.weights)

        # Матрицы весов
        FeedForwardNeuralNetwork.weights.append(np.random.normal(0.0, pow(self.inodes, -0.5), (self.inodes, self.onodes)))

        # скорость обучения
        self.lr = learningrate

    def compute_cost(self, predict, true):
            size = true.shape[0]
            cost = -(1 / size) * np.sum(true * np.log(predict) + (1 - true) * np.log(1 - predict))
            return cost
    
    # high learning rate leads to issues 

    


    def AddHiddenLayer(self, functionActivation, countNeurons):
        self.hnodes.append(Layer.Layer(functionActivation, countNeurons))
        self.weights.clear()
        if self.optFunc == 'Momentum':
            self.v.clear()
        elif self.optFunc == 'RMSprop':
            self.s.clear()    
    
        # Update quantity weights
        FeedForwardNeuralNetwork.weights.append(np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes[0].neurons, self.inodes)))
        
        if len(self.hnodes) > 1:
            for i in range(0, len(self.hnodes) - 1):
                FeedForwardNeuralNetwork.weights.append(np.random.normal(0.0, pow(self.hnodes[i].neurons, -0.5), (self.hnodes[i + 1].neurons, self.hnodes[i].neurons)))
        
        FeedForwardNeuralNetwork.weights.append(np.random.normal(0.0, pow(self.hnodes[-1].neurons, -0.5), (self.onodes, self.hnodes[-1].neurons)))
        if self.optFunc == 'Momentum':
            self.v = Optimizers.initilization_momentum(FeedForwardNeuralNetwork.weights)
        elif self.optFunc == 'RMSprop':
            self.s = Optimizers.initilization_RMS(FeedForwardNeuralNetwork.weights)

    def train(self, inputs_list, targets_list):
        
        #Считывание данных
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # Вычисление значения нейронов
        inputValue = np.dot(FeedForwardNeuralNetwork.weights[0], inputs)
        outputValue = Layer.Layer.activation(self.hnodes[0], inputValue) # 1 arg - choose layer; 2 arg - value passing 

        FeedForwardNeuralNetwork.valueLayers.append(outputValue)

        if len(FeedForwardNeuralNetwork.weights) > 1:
            for i in range(1, len(self.hnodes) + 1):
                inputValue = np.dot(FeedForwardNeuralNetwork.weights[i], outputValue)
                outputValue = Layer.Layer.activation(self.hnodes[i - 1], inputValue)
                FeedForwardNeuralNetwork.valueLayers.append(outputValue)
        
        cost = self.compute_cost(FeedForwardNeuralNetwork.valueLayers[-1], targets)

        # Вычисление ошибки
        FeedForwardNeuralNetwork.valueErrors.append(FeedForwardNeuralNetwork.valueLayers[-1] - targets)

        if len(self.hnodes) > 0:
            for i in range(len(self.hnodes), 0, -1):
                errors = np.dot(FeedForwardNeuralNetwork.weights[i].T, FeedForwardNeuralNetwork.valueErrors[-1])
                FeedForwardNeuralNetwork.valueErrors.append(errors)
        
        

        # Обновление весов
        #TODO снести это
        #Обновление весов в 67 строке и 69 надо поменять
        # 1. TODO grad отдельно
        # 2. TODO some optimizers, for example???
        # 3. TODO подумать стоить ли как нибудь рассчитать градиент, мб как то можно меньше аргументов передевать 
        if self.optFunc == 'SGD':
            FeedForwardNeuralNetwork.weights = Optimizers.SGD(self.lr, FeedForwardNeuralNetwork.weights, FeedForwardNeuralNetwork.valueErrors, FeedForwardNeuralNetwork.valueLayers, len(self.hnodes), inputs)
        elif self.optFunc == 'Momentum':   
            temp = Optimizers.Momentum(self.lr, FeedForwardNeuralNetwork.weights, FeedForwardNeuralNetwork.valueErrors, FeedForwardNeuralNetwork.valueLayers, len(self.hnodes), inputs, self.v)
            FeedForwardNeuralNetwork.weights = temp[0]
            self.v = temp[1]
        elif self.optFunc == 'RMSprop':
            temp = Optimizers.RMSprop(self.lr, FeedForwardNeuralNetwork.weights, FeedForwardNeuralNetwork.valueErrors, FeedForwardNeuralNetwork.valueLayers, len(self.hnodes), inputs, self.s)
            FeedForwardNeuralNetwork.weights = temp[0]
            self.s = temp[1]
        FeedForwardNeuralNetwork.valueLayers.clear()
        FeedForwardNeuralNetwork.valueErrors.clear()
        
        return cost

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        inputValue = np.dot(FeedForwardNeuralNetwork.weights[0], inputs)
        outputValue = Layer.Layer.activation(self.hnodes[0], inputValue)
        FeedForwardNeuralNetwork.valueLayers.append(outputValue)

        if len(FeedForwardNeuralNetwork.weights) > 1:
            for i in range(1, len(self.hnodes) + 1):
                inputValue = np.dot(FeedForwardNeuralNetwork.weights[i], outputValue)
                outputValue = Layer.Layer.activation(self.hnodes[i - 1], inputValue)
                FeedForwardNeuralNetwork.valueLayers.append(outputValue)
        finalOutput = FeedForwardNeuralNetwork.valueLayers[-1]
        FeedForwardNeuralNetwork.valueLayers.clear()
        return finalOutput
    
    
