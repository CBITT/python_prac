from pybrain.structure import FeedForwardNetwork
n = FeedForwardNetwork()


#construct input output layers
from pybrain.structure import LinearLayer, SigmoidLayer
inLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)

#add layers to the network
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

#connecting neurons from input to output
from pybrain.structure import FullConnection
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

#add connections to the network
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)

n.sortModules()


#using datasets



#test it
print (n.activate([1,2]), "-> activated")
print (in_to_hidden.params, "-> in_to_hidden.params")
print(hidden_to_out.params, "-> hiddent_to_out.params")
print(n.params, "-> n.params")




