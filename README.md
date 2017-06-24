# DataMining
Data mining project - image recognition
## Idea
 - find clusters of similar images
 - find similar objects on the image and add tags of it

## Technologies
* Python
* TensorFlow
* Convolutional neural network
* Image data - [http://image-net.org/](http://image-net.org/)

### CIFAR-10 image classification
- Download the (Python version of the) [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and extract it in a `cifar-10` top-level directory (it is in the `.gitignore`, no worries).
- Run `python cifar_10_test.py`:
    - reads the dataset
    - runs a classifier returning random classes
    - runs a classifier returning the most common class based only on a single byte (single color channel of a single pixel) x 25 times
    - displays a random image and its label (from the first batch)
