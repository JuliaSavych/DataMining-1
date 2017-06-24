import pickle
import numpy as np
from matplotlib import pyplot
import random

IMAGE_SIDE_SIZE = 32

def readFile(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

# row is a single row of 3072 uint8 values - 3 x 1024 for R, G and B.
# Each of these 1024 represent the values fotr the 32x32 image in a row-major order
# Returns 32 x 32 x 3 matrix
def cifarImageToRGBMatrix(row):
  RGBRows = np.split(row, 3)
  RGBArrays = np.dstack((RGBRows[0], RGBRows[1], RGBRows[2]))[0]
  splitIndices = np.arange(IMAGE_SIDE_SIZE, IMAGE_SIDE_SIZE * IMAGE_SIDE_SIZE - 1, IMAGE_SIDE_SIZE)
  return np.split(RGBArrays, splitIndices)

# RGBMatrix is NxMx3 or NxMx4 with uint8 or float 0..1 values
def showImage(RGBMatrix, label = 'Unknown'):
  # each image pixel will be displayed by a 2x2 square
  ax = pyplot.figure(figsize = (2,2))
  pyplot.imshow(RGBMatrix)
  ax.suptitle('Class: ' + label)
  pyplot.show()

if __name__ == "__main__":
  print('Reading CIFAR-10 data batch 1')

  batch1 = readFile('./cifar-10/data_batch_1')
  batch_meta = readFile('./cifar-10/batches.meta')

  label_names = batch_meta[b'label_names']

  # some random image to display
  imageI = random.randint(0, 9999)

  image = batch1[b'data'][imageI]
  imageMatrix = cifarImageToRGBMatrix(image)
  imageLabel = batch1[b'labels'][imageI]
  imageHumanLabel = label_names[imageLabel].decode("utf-8")

  print('Displaying image index: ', imageI)
  showImage(imageMatrix, imageHumanLabel)
