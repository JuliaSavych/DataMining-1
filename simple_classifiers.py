import random

def argmax(iterable):
  return max(enumerate(iterable), key=lambda x: x[1])[0]

class RandomClassifier:
  def __init__(self, numberOfClasses):
    self.numberOfClasses = numberOfClasses

  def train(self, batch_data, batch_labels):
    return

  def classify(self, batch_data):
    classifications = []
    for item in batch_data:
      classifications.append(random.randint(0, (self.numberOfClasses - 1)))
    return classifications


class SingleByteClassifier:
  def __init__(self, numberOfClasses, byteIndex):
    self.byteIndex = byteIndex
    self.classesByByteValue = []
    for byteValue in range(256):
      self.classesByByteValue.append([])
      for classI in range(numberOfClasses):
        self.classesByByteValue[byteValue].append(0)

  def train(self, batch_data, batch_labels):
    for i in range(len(batch_data)):
      data = batch_data[i]
      label = batch_labels[i]
      self.classesByByteValue[data[self.byteIndex]][label] += 1

  def classify(self, batch_data):
    classifications = []
    for item in batch_data:
      classifications.append(argmax(self.classesByByteValue[item[self.byteIndex]]))
    return classifications