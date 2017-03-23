import os
from random import randint, shuffle
import matplotlib.pyplot as plt
from numpy import *
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import SigmoidLayer, SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml import NetworkReader
from pybrain.tools.xml import NetworkWriter
from pybrain.utilities import percentError
from pylab import imshow
from scipy import io
from pybrain.supervised.trainers import BackpropTrainer
from PIL import Image
from scipy import io, ndimage


def plotData(image):
    '''plots the input data '''
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    plt.imshow(image.T, cmap='Greys')
    plt.show()


# load data
data = io.loadmat('ex4data1.mat')
size = (20, 20)

X = data['X']
Y = data['y']


Y[Y == 10] = 0  # 0 has the 10th position, this line gives it the 0th position

num_of_labels = unique(Y).size  # gets your 10 labels/outputs

im = Image.open("200x200.png")

# convert to numpy array
if (len(shape(im)) == 3):
    imA = asarray(im, dtype="float")[:, :, 1]
else:
    imA = asarray(im, dtype="float")

# transform pixel values from 0 to 1 and invert and convert to PIL image
imA = (imA - amin(imA)) / (amax(imA) - amin(imA))
imA = 1 - imA

im1 = asarray(imA, dtype="float")
im1 = ndimage.grey_dilation(im1, size=(5, 5))

im1 = Image.fromarray(im1)
box = (im1).getbbox()
im2 = im1.crop(box)


im3 = im2.resize(size)
im3 = asarray(im3, dtype="float")

im3 = 1 - im3.T
im3 = uint8(im3)
plotData(im3)

# build the dataset
num_of_examples, size_of_example = X.shape
# convert the test data to one of many (10)

def convert_to_one_of_many(Y):
    # converts Y to one of many types
    # or one output per label
    rows, cols = Y.shape
    classes = unique(Y).size  # should get 10 classes
    newY = zeros((rows, classes))

    for i in range(0, rows):
        newY[i, Y[i]] = 1

    return newY

Y = convert_to_one_of_many(Y)

# separating training and test data sets

X1 = hstack((X, Y)) # puts into a single one dimensional array
shuffle(X1) # shuffles the data

X = X1[:, 0:size_of_example]
Y = X1[:, size_of_example: X1.shape[1]]

dSet = ClassificationDataSet(size_of_example, num_of_labels)
for k in xrange(len(X)):
        dSet.addSample(X.ravel()[k], Y.ravel()[k])

test_data, train_data = dSet.splitWithProportion(0.3)

train_data._convertToOneOfMany()
test_data._convertToOneOfMany()
data_split = int(num_of_examples * 0.7)

# setting the field names
train_data.setField('input', X[0:data_split, :])
train_data.setField('target', Y[0:data_split, :])

#print train_data.getField(X[0:cutoff, :])

for i in range(data_split, num_of_examples):
    test_data.addSample(X[i, :], Y[i, :])

test_data.setField('input', X[data_split:num_of_examples, :])
test_data.setField('target', Y[data_split:num_of_examples, :])


if os.path.isfile('dig_img_from_dir.xml'):
    net = NetworkReader.readFrom('dig_img_from_dir.xml')
    net.sorted = False
    net.sortModules()
else:

    net = buildNetwork(size_of_example, size_of_example / 2, num_of_labels, bias=True, hiddenclass=SigmoidLayer,
                       outclass=SoftmaxLayer)
    net.sortModules()

test_index = randint(0, X.shape[0])
test_input = X[test_index, :]


real_train = train_data['target'].argmax(axis=1)
real_test = test_data['target'].argmax(axis=1)

EPOCHS = 5

trainer = BackpropTrainer(net, dataset=train_data, momentum=0.3, learningrate=0.01, verbose=False)


for i in range(EPOCHS):
    # set the epochs
    trainer.trainEpochs(1)

    outputTrain = net.activateOnDataset(train_data)
    outputTrain = outputTrain.argmax(axis=1)
    trainResult = percentError(outputTrain, real_train)

    outputTest = net.activateOnDataset(test_data)
    outputTest = outputTest.argmax(axis=1)
    testResult = percentError(outputTest, real_test)
    print('training set accuracy:', 100 - trainResult, 'test set accuracy:', 100 - testResult)


X1 = im3.reshape((X.shape[1]))
prediction = net.activate(X1)
# returns the index of the highest value down the columns
p = argmax(prediction, axis=0)
NetworkWriter.writeToFile(net, 'dig_img_from_dir.xml')

# plotData(X[:, 0:sizeOfExample-1], Y, randomIndex)
print("predicted output after training is", p)