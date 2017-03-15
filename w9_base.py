from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from sklearn.datasets import load_digits
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
import os
import matplotlib.pyplot as plt
import cv2


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
t = loadImage('8x8.png')
# load the data and store
digits = load_digits()

# set y as target
X, y = digits.data, digits.target

# add the contents digits to a dataset
daSet = ClassificationDataSet(len(t), 1)
for k in xrange(len(X)):
    daSet.addSample(X.ravel()[k], y.ravel()[k])

# split the dataset into training and testing
testData, trainData = daSet.splitWithProportion(0.25)

# convert the data to binary
trainData._convertToOneOfMany()
testData._convertToOneOfMany()

# check for the save file and load
if os.path.isfile('dig.xml'):
    net = NetworkReader.readFrom('dig.xml')
else:
    # net = FeedForwardNetwork()
    net = buildNetwork(trainData.indim, 64, trainData.outdim, outclass=SoftmaxLayer)

################# this is from an old iteration will delete

# create layers for FFN
# inLayer = LinearLayer(len(t)) #sets up the number of nodes based on 'length' of the loaded image
# hiddenLayer = SigmoidLayer(len(t))
# outLayer = LinearLayer(10)#you need ten outputs - one for each digit(0,1,2,3 etc)

# add layers to FFN
# net.addInputModule(inLayer)
# net.addModule(hiddenLayer)
# net.addOutputModule(outLayer)

# create connections between the layers
# in_to_hidden = FullConnection(inLayer, hiddenLayer)
# hidden_to_out = FullConnection(hiddenLayer, outLayer)
# add connections
# net.addConnection(in_to_hidden)
# net.addConnection(hidden_to_out)

# net.sortModules()



# a test to show the digits in the dataset, try changing the 2 and it will blwo your mind
plt.gray()
plt.matshow(digits.images[1])
plt.show()

# create a backprop trainer
trainer = BackpropTrainer(net, dataset=trainData, momentum=0.1, learningrate=0.01, verbose=True)

# set the epochs
trainer.trainEpochs(50)

# print results
print 'Percent Accuracy Test dataset: ', percentError(trainer.testOnClassData(
    dataset=testData)
    , testData['class'])

trainer.train()
#net.save()

#def save(net):
    #tree = ET.ElementTree(net)
    #tree.write("dig.xml")

    #filename = "dig.xml"
    #net.image.save(filename)
NetworkWriter.writeToFile(net, 'dig.xml')