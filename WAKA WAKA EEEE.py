import numpy as np
import scipy.special
import sklearn.metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def Query(inputs_list, fst_w, snd_w):
    inputs = np.array(inputs_list, ndmin=2).T
    
    hidden_inputs = np.dot(fst_w, inputs)
    hidden_outputs = activation_function(hidden_inputs)
    
    final_inputs = np.dot(snd_w, hidden_outputs)
    final_outputs = activation_function(final_inputs)
    
    # probability = round(((np.max(final_outputs)/np.sum(final_outputs))*100), 2)
    # print(np.argmax(final_inputs),' ',probability,'%')
    
    return final_outputs

activation_function = lambda x: scipy.special.expit(x)

with open("C:\\Users\\Mvideo\\Desktop\\weights (Traffic lights)\\fst_w_mnist.csv",'r') as f:
    fst_w = np.loadtxt(f, delimiter=",")
with open("C:\\Users\\Mvideo\\Desktop\\weights (Traffic lights)\\snd_w_mnist.csv",'r') as f:
    snd_w = np.loadtxt(f, delimiter=",")

test_data_file = open("./Data/test_dataset (MNIST).csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

a = []
b = []
# precision, recall, f1 = [], [], []
# tp, fn, fp, tn = 0, 0, 0, 0
for record in test_data_list:
    all_values = record.split(',')

    if len(all_values) == 1:
        continue

    inputs = np.asfarray(all_values[1:])
    correct_label = all_values[0]
    a.append(int(correct_label))
    outputs = Query(inputs, fst_w, snd_w)
    b.append(int(np.argmax(outputs)))
    
    # if (str(np.argmax(outputs)) + str(correct_label)) == "00":
    #     tp += 1
    # if (str(np.argmax(outputs)) + str(correct_label)) == "11":
    #     tn += 1
    # if (str(np.argmax(outputs)) + str(correct_label)) == "01":
    #     fp += 1
    # if (str(np.argmax(outputs)) + str(correct_label)) == "10":
    #     fn += 1
    # precision.append(tp/(tp + fp))
    # recall.append(tp/(tp + fn))
    # f1.append(tp/(tp + ((fp + fn)/2)))
    # precision.append(precision_score(a, b))
    # recall.append(recall_score(a, b))
    # f1.append(f1_score(a, b))

# fig = plt.figure()

# ax_1 = fig.add_subplot(2, 2, 1)
# ax_2 = fig.add_subplot(2, 2, 2)
# ax_3 = fig.add_subplot(2, 2, 3)
# ax_4 = fig.add_subplot(2, 2, 4)

# ax_1.set(title = 'recall')
# ax_2.set(title = 'precision')
# ax_3.set(title = 'recall-precision curve')
# ax_4.set(title = 'f1')

# ax_1.plot(recall)
# ax_2.plot(precision)
# ax_3.plot(recall, precision)
# ax_3.set_xlabel('recall')
# ax_3.set_ylabel('precision')
# ax_4.plot(f1)

precision_1 = precision_score(a, b, average="weighted")
recall_1 = recall_score(a, b, average="weighted")
f1_1 = f1_score(a, b, average="weighted")
r = sklearn.metrics.confusion_matrix(a, b)
print(precision_1, recall_1, f1_1, end='\n')
print(r)

plt.show()