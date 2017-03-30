import numpy as np
from random import randint
import pickle
import gzip
import matplotlib.pyplot as plt
from pylab import *


# Load the MNIST dataset
with gzip.open('mnist.pkl.gz', 'r') as f:
    train_set, valid_set, test_set = pickle.load(f)

#input to tuple (x,y)
learn_data       = [(train_set[0][i], [1 if j == train_set[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(train_set[0]))]
test_data        = [(test_set[0][i], [1 if j == test_set[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(test_set[0]))]
validation_data  = [(valid_set[0][i], [1 if j == valid_set[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(valid_set[0]))]

def sigmoid( x ):
    return np.nan_to_num( 1/(1+np.exp(-x)) )
def sigmoid_deriv( x ):
    return sigmoid(x)*(1-sigmoid(x))


class QuadraticCost:

    @staticmethod
    def fn(activations, targets):
      return 0.5 * (activations - targets) ** 2

    @staticmethod
    def fn_deriv(activations, targets):
      return activations - targets

    @staticmethod
    def delta(inputs, activations, targets):
        return (activations - targets) * sigmoid_deriv(inputs)


class neuralnetwork:
    def __init__(self, shape, cost=QuadraticCost):
        """ Initialize the neural network """

        self.shape = shape
        self.number_of_layers = len(shape)
        self.cost = cost
        self.weights = [np.random.normal(0, 1 / np.sqrt(shape[i + 1]), (shape[i], shape[i + 1])) \
                        for i in range(self.number_of_layers - 1)]

        self.biases = [np.random.normal(0, 1, (shape[i])) \
                       for i in range(1, self.number_of_layers)]

    def feedforward(self, inputdata):

        self.input_to_layer = {}
        self.output_from_layer = {}

        self.input_to_layer[0] = inputdata
        self.output_from_layer[0] = np.array(inputdata)

        # Feed input through the layers
        for layer in range(1, self.number_of_layers):
            self.input_to_layer[layer] = np.dot(self.output_from_layer[layer - 1], \
                                                self.weights[layer - 1]) + self.biases[layer - 1]
            self.output_from_layer[layer] = np.array(sigmoid(self.input_to_layer[layer]))

        # Return output from last layer
        return self.output_from_layer[self.number_of_layers - 1]

    def backpropagate(self, targets):

        self.delta = {}
        self.del_cost_del_bias = {}
        self.del_cost_del_weight = {}

        self.delta[self.number_of_layers - 1] = \
            (self.cost).delta(self.input_to_layer[self.number_of_layers - 1], \
                              self.output_from_layer[self.number_of_layers - 1], targets)

        # Compute the delta's for the other layers
        for layer in np.arange(self.number_of_layers - 2, -1, -1):
            self.delta[layer] = np.dot(self.delta[layer + 1], self.weights[layer].T) * \
                                sigmoid_deriv(np.array(self.input_to_layer[layer]))

        # Compute partial derivatives of C w.r.t the biases and the weights
        for layer in np.arange(self.number_of_layers - 1, 0, -1):
            self.del_cost_del_bias[layer] = self.delta[layer]
            self.del_cost_del_weight[layer] = np.dot(self.output_from_layer[layer - 1].T, \
                                                     self.delta[layer])

        return self.del_cost_del_bias, self.del_cost_del_weight

    def train_mini_batch(self, data, rate, l2):
        """ Train the network on a mini-batch """

        # Split the data into input and output
        inputs = [entry[0] for entry in data]
        targets = [entry[1] for entry in data]

        # Feed the input through the network
        self.feedforward(inputs)
        # Propagate the error backwards
        self.backpropagate(targets)

        # Update the weights and biases
        n = len(targets)
        for layer in np.arange(1, self.number_of_layers):
            self.biases[layer - 1] -= (rate) * np.mean(self.del_cost_del_bias[layer], axis=0)
            self.weights[layer - 1] -= (rate / n) * self.del_cost_del_weight[layer] - \
                                       rate * l2 * self.weights[layer - 1]

    def stochastic_gradient_descent(self, data, number_of_epochs, mini_batch_size, \
                                    rate=1, l2=0.1, test_data=None):
        """ Train the network using the stochastic gradient descent method. """

        # For every epoch:
        for epoch in np.arange(number_of_epochs):
            # Randomly split the data into mini_batches
            np.random.shuffle(data)
            batches = [data[x:x + mini_batch_size] \
                       for x in np.arange(0, len(data), mini_batch_size)]

            for batch in batches:
                self.train_mini_batch(batch, rate, l2)

            if test_data != None:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), \
                                                    len(test_data)))

    def evaluate(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly
            evaluated. """
        count = 0
        for testcase in test_data:
            answer = np.argmax(testcase[1])
            prediction = np.argmax(self.feedforward(testcase[0]))
            count = count + 1 if (answer - prediction) == 0 else count
        return count

    def save(self, filename):
        """ Save neural network (weights) to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'biases': self.biases, 'weights': self.weights}, f)

    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Set biases and weights
        self.biases = data['biases']
        self.weights = data['weights']

mynet = neuralnetwork( [784,100,10] )
mynet.stochastic_gradient_descent( learn_data, 25, 10, 0.1, 0.001/len(train_set[0]), \
                                   test_data = validation_data )

mynet.save("MNIST-CrossEntropy-Network")
mynet = neuralnetwork( [784,30,10] )
mynet.load("MNIST-CrossEntropy-Network")

# pick random number from dataset
imgnr = np.random.randint(0,10000)
prediction = mynet.feedforward( test_set[0][imgnr] )
print("Image number {0} is a {1}, and our network predicted a {2}".format(imgnr, test_set[1][imgnr], np.argmax(prediction)))

fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].matshow( np.reshape(test_set[0][imgnr], (28,28) ), cmap=cm.gray )
ax[1].plot( prediction, lw=3 )
ax[1].set_aspect(9)
plt.show()