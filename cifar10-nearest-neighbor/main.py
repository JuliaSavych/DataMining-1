from data_readers import CfarDataReader
from nearest_neighbor import NearestNeighbor

if __name__ == "__main__":
    data_reader = CfarDataReader()
    training_data = data_reader.get_training_batches()
    test_data = data_reader.get_testing_batch()

    classifier = NearestNeighbor()
    for batch in training_data:
        classifier.train(batch[b'data'], batch[b'labels'])

    start = 0
    end = 10

    results = classifier.classify(test_data[b'data'][start:end])

    correct_labels = test_data[b'labels'][start:end]
    correct = 0
    for i, res in enumerate(results):
        if res == correct_labels[i]:
            correct += 1
    print(correct / len(results))