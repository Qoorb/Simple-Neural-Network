import numpy as np
import Model
import matplotlib.pyplot as plt

input_nodes = 784
output_nodes = 10
costs = []
learning_rate = 0.001
epochs = 10

n = Model.FeedForwardNeuralNetwork(input_nodes, output_nodes, learning_rate, optimizer='SGD')
n.AddHiddenLayer(functionActivation='sigmoid', countNeurons=500)
n.AddHiddenLayer(functionActivation='sigmoid', countNeurons=500)

# Считывание данных для тренировки
training_data_file = open("./Data/train_dataset (MNIST).csv", 'r')
training_data_list = np.array(training_data_file.readlines())
training_data_file.close()

costs = n.train(training_data_list, epochs)

# Считывание данных для тестов
test_data_file = open("./Data/test_dataset (MNIST).csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')

    if len(all_values) == 1:
        continue

    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
 
    outputs = n.query(inputs)
    label = np.argmax(outputs) 
    print(label, correct_label, sep="=")

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
# 0 = "red"
# 1 = "green"
# 2 = "Nothing"

scorecard_array = np.asarray(scorecard)
print (f"performance = {(scorecard_array.sum() / scorecard_array.size)*100} %")

iters = [i for i in range(epochs)] # iters = [0, 1]; costs = [x_1, x_2]
print(iters, costs, sep='\n')
plt.plot(iters, costs, color="blue", label="SGD")

plt.xlabel('epochs')
plt.ylabel('func_cost')
plt.legend()
plt.title('visualization of different optimizers')
# plt.imshow()
# plt.savefig()
plt.show()