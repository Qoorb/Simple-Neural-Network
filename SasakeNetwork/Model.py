import numpy as np
import scipy.special
import Layer

class FeedForwardNeuralNetwork:
    
    weights = []
    valueLayers = []
    valueErrors = []

    # Инициализация ИНС
    def __init__(self, inputnodes, outputnodes, learningrate):
        # Входные, скрытые и выходные нейроны
        self.inodes = inputnodes 
        self.hnodes = []
        self.onodes = outputnodes

        # Матрицы весов
        FeedForwardNeuralNetwork.weights.append(np.random.normal(0.0, pow(self.inodes, -0.5), (self.inodes, self.onodes)))

        # скорость обучения
        self.lr = learningrate
        
        # Функция активации - сигмоидная функция из библиотеки scipy
        self.activation_function = lambda x: scipy.special.expit(x)


    def AddHiddenLayer(self, functionActivation, countNeurons):
        self.hnodes.append(Layer.Layer(functionActivation, countNeurons))
        self.weights.clear()
        
        # Update quantity weights
        FeedForwardNeuralNetwork.weights.append(np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes[0].neurons, self.inodes)))
        
        if len(self.hnodes) > 1:
            for i in range(0, len(self.hnodes) - 1):
                FeedForwardNeuralNetwork.weights.append(np.random.normal(0.0, pow(self.hnodes[i].neurons, -0.5), (self.hnodes[i + 1].neurons, self.hnodes[i].neurons)))
        
        FeedForwardNeuralNetwork.weights.append(np.random.normal(0.0, pow(self.hnodes[-1].neurons, -0.5), (self.onodes, self.hnodes[-1].neurons)))
        

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # Вычисление значения нейронов
        inputValue = np.dot(FeedForwardNeuralNetwork.weights[0], inputs)
        outputValue = self.activation_function(inputValue)
        FeedForwardNeuralNetwork.valueLayers.append(outputValue)

        if len(FeedForwardNeuralNetwork.weights) > 1:
            for i in range(1, len(self.hnodes) + 1):
                inputValue = np.dot(FeedForwardNeuralNetwork.weights[i], outputValue)
                outputValue = self.activation_function(inputValue)
                FeedForwardNeuralNetwork.valueLayers.append(outputValue)
        
        # Вычисление ошибки
        FeedForwardNeuralNetwork.valueErrors.append(targets - FeedForwardNeuralNetwork.valueLayers[-1])

        if len(self.hnodes) > 0:
            for i in range(len(self.hnodes), 0, -1):
                errors = np.dot(FeedForwardNeuralNetwork.weights[i].T, FeedForwardNeuralNetwork.valueErrors[-1])
                FeedForwardNeuralNetwork.valueErrors.append(errors)
        
        # Обновление весов
        if len(FeedForwardNeuralNetwork.weights) > 1:
            for i in range(len(self.hnodes), 0, -1):
                FeedForwardNeuralNetwork.weights[i] += self.lr * np.dot((FeedForwardNeuralNetwork.valueErrors[-(i+1)] * FeedForwardNeuralNetwork.valueLayers[i] * (1.0 - FeedForwardNeuralNetwork.valueLayers[i])), np.transpose(FeedForwardNeuralNetwork.valueLayers[i - 1]))

        FeedForwardNeuralNetwork.weights[0] += self.lr * np.dot((FeedForwardNeuralNetwork.valueErrors[-1] * FeedForwardNeuralNetwork.valueLayers[0] * (1.0 - FeedForwardNeuralNetwork.valueLayers[0])), np.transpose(inputs))

        FeedForwardNeuralNetwork.valueLayers.clear()
        FeedForwardNeuralNetwork.valueErrors.clear()

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        inputValue = np.dot(FeedForwardNeuralNetwork.weights[0], inputs)
        outputValue = self.activation_function(inputValue)
        FeedForwardNeuralNetwork.valueLayers.append(outputValue)

        if len(FeedForwardNeuralNetwork.weights) > 1:
            for i in range(1, len(self.hnodes) + 1):
                inputValue = np.dot(FeedForwardNeuralNetwork.weights[i], outputValue)
                outputValue = self.activation_function(inputValue)
                FeedForwardNeuralNetwork.valueLayers.append(outputValue)
        finalOutput = FeedForwardNeuralNetwork.valueLayers[-1]
        FeedForwardNeuralNetwork.valueLayers.clear()
        return finalOutput