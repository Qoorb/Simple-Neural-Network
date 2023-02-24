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

# myFileList = createFileList('./Samples/train (only green)')

# for file in myFileList:
#     print(file)
#     img_file = Image.open(file)

#     # Параметры исходного изображения
#     width, height = img_file.size
#     format = img_file.format
#     mode = img_file.mode

#     img_grey = img_file.convert('L')
#     img_grey = img_grey.resize((32, 32))

#     # Конвертируем в формат .csv
#     value = np.asarray(img_grey.getdata(), dtype=int).reshape((img_grey.size[1], img_grey.size[0]))
    
#     value = value.flatten()
#     value = np.insert(value, 0, 1)

#     with open("train_data.csv", 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(value)



<<<<<<< HEAD
myFileList = createFileList('./Samples/GC_1')

data = []
=======
myFileList = createFileList('./Samples/test (red)')

# data = []
>>>>>>> 093f8e6b8e32fa55868c1353fa45e32d868e0d1e
for file in myFileList:
    img_file = Image.open(file)

    img_hsv = img_file.convert('HSV')
    img_hsv = img_hsv.resize((32, 32))

    # print(img_hsv)
    # Конвертируем в формат .csv
<<<<<<< HEAD
    data.append(1)
=======
    # data.append(0)
    # value = np.asarray((np.array([0]) + np.array(img_hsv.getdata(), dtype=int))) # 2-D array
>>>>>>> 093f8e6b8e32fa55868c1353fa45e32d868e0d1e
    value = np.asarray(img_hsv.getdata(), dtype=int)
    # print(value)
    # value = np.asarray(img_hsv.getdata(), dtype=int)
    # for i in range(img_hsv.size[0] * img_hsv.size[1]):
    #   data.append([value[i][0], value[i][1], value[i][2]])
    value = np.insert(value, 0, 0)
    value = value.flatten()
    # data = np.insert(data, 0, 1)
    # 0 = Red
    # 1 = Green

<<<<<<< HEAD
    with open("new_train_data.csv", 'a') as f:
=======
    with open("TEST_DATA_Q.csv", 'a') as f:
>>>>>>> 093f8e6b8e32fa55868c1353fa45e32d868e0d1e
        writer = csv.writer(f)
        writer.writerow(value)