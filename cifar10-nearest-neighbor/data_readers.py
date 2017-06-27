import pickle


class CfarDataReader:
    def __init__(self):
        pass

    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_training_batches(self):
        batches = []
        for i in range(1, 6):
            batches.append(self._unpickle('../cifar-10/data_batch_%d' % i))
        return batches

    def get_testing_batch(self):
        return self._unpickle('../cifar-10/test_batch')
