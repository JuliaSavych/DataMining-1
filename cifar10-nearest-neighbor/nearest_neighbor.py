import math
import numpy as np

class NearestNeighbor:
    def __init__(self):
        self.batches = []
        self.labels = []

    def train(self, batches, labels):
        for batch in batches:
            self.batches.append(batch)
        for label in labels:
            self.labels.append(label)

    def classify(self, items):
        classifications = []
        i = 1
        print('All items: ', len(items))
        for item in items:
            index = self._getNearest(item)
            cls = self.labels[index]
            print('item: ', i, 'of ', len(items))
            i = i + 1
            classifications.append(cls)
        return classifications

    def _getNearest(self, item):
        min = math.inf
        min_index = 0
        for i, batch in enumerate(self.batches):
            dist = np.sqrt(np.sum(np.square(item - batch)))
            if dist < min:
                min = dist
                min_index = i
        return min_index
