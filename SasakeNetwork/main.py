import numpy as np
import Model
import matplotlib.pyplot as plt

input_nodes = 784
output_nodes = 10
costs = []
itr = []
learning_rate = 0.001

n = Model.FeedForwardNeuralNetwork(input_nodes, output_nodes, learning_rate, optimizer='RMSprop')
n.AddHiddenLayer(functionActivation='sigmoid', countNeurons=500)
n.AddHiddenLayer(functionActivation='sigmoid', countNeurons=500)

# Считывание данных для тренировки
training_data_file = open("./Data/train_dataset (MNIST).csv", 'r')
training_data_list = np.array(training_data_file.readlines())
training_data_file.close()

epochs = 1
for e in range(epochs):
    for i in range(len(training_data_list)):
        all_values = list(training_data_list[i].split(','))

        if len(all_values) == 1:
            continue

        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01

        targets[int(all_values[0])] = 0.99

        cost = n.train(inputs, targets)
        if i % 5 == 0:
            costs.append(cost)
            itr.append(i)
            if i % 100 == 0 :
                print('cost of iteration______{}______{}'.format(i, cost))

    print(f"Эпоха: {e + 1}")



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


plt.plot(itr, costs,color="black",label="rmsprop")
plt.xlabel('num_iter')
plt.ylabel('cost')
plt.legend()
plt.title('visualization of different optimizers')
plt.show()