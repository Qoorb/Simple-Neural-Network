# import numpy as np 

# def sigmoid(x):
#     return 1 / (1 + np.exp(-(x)))

# def mse_loss(true, pred):
#     return 0.5 * (pred - true)**2

# class NeuralNetwork:
#     def __init__(self, layers, neurons):
#         self.countLayers = layers
#         self.neurons = neurons
#         self.weights = [([1] * (self.neurons[x] * self.neurons[x + 1])) for x in range(self.countLayers - 1)]
#         self.bias = [1] * sum(self.neurons[1:])
#     def feedForward(self, data):
#         pass
#     def train(self):
#         pass

# l = int(input("Введите количество слоев: "))
# n = [int(input(f"Количество нейронов на {x + 1} слое: ")) for x in range(l)]

# network = NeuralNetwork(l, n)
# print(network.weights, network.neurons, sep='\n')

from PIL import Image
import numpy as np
import sys
import os
import csv

# Запуск = смерть
def createFileList(myDir, format='.jpg'):
  fileList = []
  for root, dirs, files in os.walk(myDir, topdown=False):
      for name in files:
         if name.endswith(format):
              fullName = os.path.join(root, name)
              fileList.append(fullName)
  return fileList

myFileList = createFileList('./Samples/train')

for file in myFileList:
    print(file)
    img_file = Image.open(file)

    # Параметры исходного изображения
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Если надо будет GrayScale
    img_grey = img_file.convert('L')
    img_grey = img_grey.resize((32, 32))
    # img_grey.show()
    # Конвертируем в формат .csv
    value = np.asarray(img_grey.getdata(), dtype=int).reshape((img_grey.size[1], img_grey.size[0]))
    print(value)

    value = value.flatten()
    print(value)
    with open("train_data.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)



myFileList = createFileList('./Samples/test')

for file in myFileList:
    print(file)
    img_file = Image.open(file)

    # Параметры исходного изображения
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Если надо будет GrayScale
    img_grey = img_file.convert('L')
    img_grey = img_grey.resize((32, 32))

    # img_grey.show()

    # Конвертируем в формат .csv
    value = np.asarray(img_grey.getdata(), dtype=int).reshape((img_grey.size[1], img_grey.size[0]))
    print(value)
    value = value.flatten()
    print(value)
    with open("test_data.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)