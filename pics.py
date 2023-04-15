import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from PIL import Image
import numpy as np
import cv2
import os

def createFileList(myDir, format='.jpg'):
  fileList = []
  for root, dirs, files in os.walk(myDir, topdown=False):
      for name in files:
         if name.endswith(format):
              fullName = os.path.join(root, name)
              fileList.append(fullName)
  return fileList

myFileList = createFileList('./Samples/pics (MNIST)')

fig = plt.figure()

ax_1 = fig.add_subplot(2, 2, 1)
ax_2 = fig.add_subplot(2, 2, 2)
ax_3 = fig.add_subplot(2, 2, 3)
ax_4 = fig.add_subplot(2, 2, 4)

image_1 = plt.imread(myFileList[0])
image_2 = plt.imread(myFileList[1])
image_3 = plt.imread(myFileList[2])
image_4 = plt.imread(myFileList[3])

# image_1 = cv2.resize(image_1, (32, 32))
# image_2 = cv2.resize(image_2, (32, 32))
# image_3 = cv2.resize(image_3, (32, 32))
# image_4 = cv2.resize(image_4, (32, 32))

ax_1.imshow(image_1)
ax_2.imshow(image_2)
ax_3.imshow(image_3)
ax_4.imshow(image_4)

# ax_1.get_xaxis().set_visible (False)
# ax_1.get_yaxis().set_visible (False)
# ax_2.get_xaxis().set_visible (False)
# ax_2.get_yaxis().set_visible (False)
# ax_3.get_xaxis().set_visible (False)
# ax_3.get_yaxis().set_visible (False)
# ax_4.get_xaxis().set_visible (False)
# ax_4.get_yaxis().set_visible (False)

ax_1.set_xlabel('2')
ax_2.set_xlabel('9')
ax_3.set_xlabel('5')
ax_4.set_xlabel('4')

plt.show()