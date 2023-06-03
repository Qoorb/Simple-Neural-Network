import numpy as np
import Optimizers
import Layer

class FeedForwardNeuralNetwork:
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

    def train(self, data_list, epochs):
        res_cost = []
        for e in range(epochs):
            сosts = []
            for minie in range(len(data_list)):
                all_values = list(data_list[minie].split(','))

                if len(all_values) == 1:
                   continue

                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                targets = np.zeros(self.onodes) + 0.01

                targets[int(all_values[0])] = 0.99

                #Считывание данных
                inputs = np.array(inputs, ndmin=2).T
                targets = np.array(targets, ndmin=2).T
        
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
                # 1. TODO grad отдельно
                # 2. TODO подумать стоить ли как нибудь рассчитать градиент, мб как то можно меньше аргументов передевать 
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
                
                сosts.append(cost)

                if minie % 100 == 0:
                    print('cost of iteration______{}______{}'.format(minie, cost))
            print(f"Epoch______{e + 1}______{sum(сosts) / len(сosts)}")
            res_cost.append(sum(сosts) / len(сosts))

        
        return res_cost

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