# importing MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("- MNIST dataset imported\n")

# output the format of train, test, and validation data
print("- Trainning data and label dimension:")
print(mnist.train.images.shape, mnist.train.labels.shape)
print("- Testing data and label dimension:")
print(mnist.test.images.shape, mnist.test.labels.shape)
print("- Validation data and label dimension:")
print(mnist.validation.images.shape, mnist.validation.labels.shape)
print