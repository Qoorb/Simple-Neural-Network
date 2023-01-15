import numpy as np
import scipy.special
import matplotlib.pyplot

class neuralNetwork:
    # Инициализация ИНС
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Входные, скрытые и выходные нейроны
        self.inodes = inputnodes 
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Матрицы весов
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # скорость обучения
        self.lr = learningrate
        
        # Функция активации - сигмоидная функция из библиотеки scipy
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # Вычисление значения нейронов
        hidden_inputs = np.dot(np.squeeze(self.wih), np.squeeze(inputs))
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # Вычисление ошибки
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors) 
        
        # Обновление весов
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

input_nodes = 1024
hidden_nodes = 1024
output_nodes = 1024

learning_rate = 0.8

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# Считывание данных для тренировки
training_data_file = open("train_data.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')

        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        n.train(inputs, targets)
        pass
    pass


# Считывание данных для тестов
test_data_file = open("test_data.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')

    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    outputs = n.query(inputs)
    label = np.argmax(outputs)

    # Добавление в список правильных и неправильных ответов
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# Оценка ИНС
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)