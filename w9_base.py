from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from sklearn.datasets import load_digits
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
NUM_EPOCHS = 50

# read image with cv2
def loadImage(path):
    im = cv2.imread(path)
    return flatten(im)


# flatten the image
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


# pass image to store image a store it as t


#t = loadImage('8x8.png')

#load the data and store
digits = load_digits()

def normalizeImage():
    image = Image.open("8x8.png")
    image = image.convert('LA')
    image.thumbnail((8,8),Image.ANTIALIAS)
    image.save("numberOut.png")


#set y as target
X, y = digits.data, digits.target


#add the contents of digits to a dataset
daSet = ClassificationDataSet(64, 1)
for k in xrange(len(X)):
    daSet.addSample(X.ravel()[k], y.ravel()[k])

#split the dataset into training and testing
testData, trainData = daSet.splitWithProportion(0.40)

#convert the data into 10 separate digits
trainData._convertToOneOfMany()
testData._convertToOneOfMany()


#check for the save file and load
if os.path.isfile('dig.xml'):
    net = NetworkReader.readFrom('dig.xml')
    net.sorted = False
    net.sortModules()
else:
    # net = FeedForwardNetwork()
    net = buildNetwork(64, 37,10, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer, bias=True)

# create a backprop trainer
trainer = BackpropTrainer(net, dataset=trainData, momentum=0.0, learningrate=0.01,weightdecay= 0.01, verbose=True)

trainer.trainUntilConvergence()

print(trainData.indim)

print(testData.indim)



#a test to show the digits in the dataset, try changing the 2 and it will blwo your min
"""plt.gray()
plt.matshow(digits.images[2])
plt.show()"""




#set the epochs
#trainer.trainEpochs(5)
NetworkWriter.writeToFile(net, 'dig.xml')



#print net.activate(t)


#print results
#print 'Percent Error dataset: ', percentError(trainer.testOnClassData(
#    dataset=testData)
#    , testData['class'])

exit(0)
