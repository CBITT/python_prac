# Inputs	Output
# 0	0 1   	0
# 1	1 1   	1
# 1	0 1   	1
# 0	1 1   	0


import numpy as np #linear algebra library

#sigmoid function

def nonlin(x,deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

#input dataset

X = np.array([  [0,0,1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]])

#output dataset

y = np.array([[0,0,1,1]]).T   #.T is the transpose function.
                                #  makes the line stand up in one column

# seed random numbers to make calculation
np.random.seed(1)
# we only have two layers (input and output) so we only need one matrix of weight (3,1)
# 3 -> number of inputs, 1-> number of outputs

syn0 = 2*np.random.random((3,1)) - 1

#syn0 is the only value gets saved

#begin the training code
for iter in xrange(10000):

    # forward propagation
    # l0 is the first layer data. x is the collection of the 4 rows
    # process all rows at the same time (in this case 4 different l0)
    l0 = X

    # next is the prediction set. try to predict the output given the input
    #
    l1 = nonlin(np.dot(l0, syn0))

# what did I miss?
    l1_error = y - l1

# multiple how much we missed by the slope of the sigmoid at the value in l1

l1_delta = l1_error * nonlin(l1,True)

#update weights

syn0 += np.dot (l0.T, l1_delta)

print "Output after training:"
print  l1



