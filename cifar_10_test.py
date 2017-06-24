import pickle
import numpy as np
from matplotlib import pyplot
import random
from simple_classifiers import RandomClassifier, SingleByteClassifier

IMAGE_SIDE_SIZE = 32
BATCH_FILENAMES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

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
  ax = pyplot.figure(figsize = (1.5,1.5))
  pyplot.imshow(RGBMatrix)
  ax.suptitle(label)
  pyplot.show()

def trainAndClassify(batches, test_batch, classifier):
  for batch in batches:
    classifier.train(batch[b'data'], batch[b'labels'])
  return classifier.classify(test_batch[b'data'])

def calculateAccuracy(results, test_batch):
  correct = 0
  for i, res in enumerate(results):
    if res == test_batch[b'labels'][i]:
      correct += 1
  return (correct / len(test_batch[b'labels']))

def showRandomImage(batch_data, batch_labels, label_names):
  imageI = random.randint(0, len(batch_data) - 1)
  image = batch_data[imageI]
  imageMatrix = cifarImageToRGBMatrix(image)
  imageLabel = batch_labels[imageI]
  imageHumanLabel = label_names[imageLabel].decode("utf-8")
  print('\nDisplaying random image index: ', imageI)
  showImage(imageMatrix, imageHumanLabel)

if __name__ == "__main__":
  batches = []

  for batchFilename in BATCH_FILENAMES:
    print('Reading file ' + batchFilename)
    batch = readFile('./cifar-10/' + batchFilename)
    batches.append(batch)

  test_batch = readFile('./cifar-10/test_batch')

  resultsRandom = trainAndClassify(batches, test_batch, RandomClassifier(10))
  accuracyRandom = calculateAccuracy(resultsRandom, test_batch)
  print('\nAccuracy of random classifier: ' + str(accuracyRandom))

  for i in range(25):
    resultsSingleByte = trainAndClassify(batches, test_batch, SingleByteClassifier(10, i))
    accuracySingleByte = calculateAccuracy(resultsSingleByte, test_batch)
    print('Accuracy of single byte classifier for byte ' + str(i) + ': ' + str(accuracySingleByte))

  batch_meta = readFile('./cifar-10/batches.meta')
  label_names = batch_meta[b'label_names']
  # some random image to display
  showRandomImage(batches[0][b'data'], batches[0][b'labels'], label_names)
