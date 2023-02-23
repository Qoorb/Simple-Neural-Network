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

myFileList = createFileList('./Samples/RC_1')

for file in myFileList:
    img = Image.open(file)
 
    # 2. Convert image to NumPy array
    arr = np.asarray(img)

    # 3. Convert 3D array to 2D list of lists
    lst = []
    for row in arr:
        tmp = []
        for col in row:
            tmp.append(str(col))
        lst.append(tmp)

    with open('my_file.csv', 'w') as f:
        for row in lst:
            f.write(','.join(row) + '\n')