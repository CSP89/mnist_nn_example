# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
import numpy
import scipy.special
import pandas as pd
from NeuronalNetwork import NeuronalNetwork

#%%
inputNodes = 28*28
hiddenNodes = 200
outputNodes = 10

learningRate = .2

epochs = 7

n = NeuronalNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

trainData = pd.read_csv("https://pjreddie.com/media/files/mnist_train.csv", header=None)

for e in range(epochs):
    for index, allValues in trainData.iterrows():
        inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + .01
        targets = numpy.zeros(outputNodes) + .01
        targets[int(allValues[0])] = .99
        n.train(inputs,targets)
        
        pass
    
    pass


#%%
scorecard = []
testData = pd.read_csv("https://pjreddie.com/media/files/mnist_test.csv", header=None)

for index, allValues in testData.iterrows():
    
    corectLabel = int(allValues[0])
    inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
    
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    
    if(label == corectLabel):
        scorecard.append(1)
    else:
        scorecard.append(0)
        
        pass
    
    pass

scorecardArray = numpy.asarray(scorecard)
print("performance = ",'%.20f' %( scorecardArray.sum() / scorecardArray.size))