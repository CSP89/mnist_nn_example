# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
import numpy
import scipy.special
import pandas as pd

#%%
class NeuronalNetwork:
    iNodes: int
    hNodes: int
    oNodes: int

    wih: [[float]]
    who: [[float]]

    def activationFunction(self, x): return scipy.special.expit(x)

    def __init__(self, inputNodes: int, hiddenNodes: int, outputNodes: int, lerningRate: float):
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.lr = lerningRate

        self.wih = (numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes)))
        self.who = (numpy.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes)))

        pass

    def train(self, inputList, targetList): 
        inputs = numpy.array(inputList, ndmin=2).T
        targets = numpy.array(targetList, ndmin=2).T

        hiddenInputs = self.hiddenInputs(inputs)
        hiddenOutputs = self.hiddenOutputs(hiddenInputs)

        finalInputs = self.finalInputs(hiddenOutputs)
        finalOutputs = self.finalOutputs(finalInputs)

        outputErrors = self.outputErrors(targets, finalOutputs)
        hiddenErrors = self.hiddenErrors(outputErrors)

        self.who += self.lr * numpy.dot(
            (outputErrors * finalOutputs * (1.0 - finalOutputs)),
            numpy.transpose(hiddenOutputs)
        )
        
        self.wih += self.lr * numpy.dot(
            (hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)),
            numpy.transpose(inputs)
        )

        pass

    def hiddenInputs(self, inputs): return numpy.dot(self.wih, inputs)
    def hiddenOutputs(self, hiddenInputs): return self.activationFunction(hiddenInputs)
    def hiddenErrors(self, outputErrors): return numpy.dot(self.who.T, outputErrors)

    def finalInputs(self, hiddenOutputs): return numpy.dot(self.who, hiddenOutputs)
    def finalOutputs(self, finalInputs): return self.activationFunction(finalInputs)

    def outputErrors(self, targets, finalOutputs): return targets - finalOutputs

    def query(self, inputList):
        hiddenInputs = self.hiddenInputs(inputList)
        hiddenOutputs =  self.hiddenOutputs(hiddenInputs)

        finalInputs = self.finalInputs(hiddenOutputs)
        return self.finalOutputs(finalInputs)

    pass

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

#%%
allValues = testData.iloc[9]
inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01

outputs = n.query(inputs)

numpy.argmax(outputs)
