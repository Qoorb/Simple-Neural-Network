from PIL import Image
import numpy as np
import sys
import os
import csv

def createFileList(myDir, format='.jpg'):
  fileList = []
  for root, dirs, files in os.walk(myDir, topdown=False):
      for name in files:
         if name.endswith(format):
              fullName = os.path.join(root, name)
              fileList.append(fullName)
  return fileList

myFileList = createFileList('./Samples/train (only green)')

for file in myFileList:
    print(file)
    img_file = Image.open(file)

    # Параметры исходного изображения
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    img_grey = img_file.convert('L')
    img_grey = img_grey.resize((32, 32))

    # Конвертируем в формат .csv
    value = np.asarray(img_grey.getdata(), dtype=int).reshape((img_grey.size[1], img_grey.size[0]))
    
    value = value.flatten()
    value = np.insert(value, 0, 1)

    with open("train_data.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)



myFileList = createFileList('./Samples/test (only green)')

for file in myFileList:
    print(file)
    img_file = Image.open(file)

    # Параметры исходного изображения
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    img_grey = img_file.convert('L')
    img_grey = img_grey.resize((32, 32))

    # Конвертируем в формат .csv
    value = np.asarray(img_grey.getdata(), dtype=int).reshape((img_grey.size[1], img_grey.size[0]))

    value = value.flatten()
    value = np.insert(value, 0, 1)

    with open("test_data.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)