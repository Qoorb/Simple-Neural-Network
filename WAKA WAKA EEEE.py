import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def Query(inputs_list, fst_w, snd_w):
    inputs = np.array(inputs_list, ndmin=2).T
    
    hidden_inputs = np.dot(fst_w, inputs)
    hidden_outputs = activation_function(hidden_inputs)
    
    final_inputs = np.dot(snd_w, hidden_outputs)
    final_outputs = activation_function(final_inputs)
    
    probability = round(((np.max(final_outputs)/np.sum(final_outputs))*100), 2)

    print(np.argmax(final_inputs),' ',probability,'%')

activation_function = lambda x: scipy.special.expit(x)

with open('Weights (Waka Waka)/fst_w.csv','r') as f:
    fst_w = np.loadtxt(f, delimiter=",")
with open('Weights (Waka Waka)/snd_w.csv','r') as f:
    snd_w = np.loadtxt(f, delimiter=",")

test_data_file = open("./Data/TEST_DATA_Q.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

for record in test_data_list:
    all_values = record.split(',')

    if len(all_values) == 1:
        continue

    inputs = np.asfarray(all_values[1:])

    print(all_values[0], end="=")
    outputs = Query(inputs, fst_w, snd_w)