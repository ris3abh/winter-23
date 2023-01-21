from abc import ABC, abstractmethod 
import numpy as np

class Layer (ABC) :
    def init (self): 
        self . prevIn = [] 
        self. prevOut=[]

    def setPrevIn(self ,dataIn): 
        self . prevIn = dataIn

    def setPrevOut( self , out ): 
        self . prevOut = out

    def getPrevIn( self ):
        return self . prevIn

    def getPrevOut( self ):
        return self . prevOut

    @abstractmethod
    def forward(self ,dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward( self , gradIn ):
        pass

## input layer z scores all the input's
class inputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn)
        self.stdX = np.std(dataIn)

    def forward(self, dataIn):
        self.prevIn = dataIn
        dataOut =  (dataIn - self.meanX)/self.stdX
        self.setprevOut = dataOut
        return dataOut

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

## linear layer is just linear activation or identity actiavation
class  LinearLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.dataIn = dataIn
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut =  dataIn
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

## implementing logistic function as the activation function
class logisticSigmoidLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.dataIn = dataIn

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut =  1/(1 + np.exp(-dataIn))
        self.setPrevOut = dataOut
        return dataOut

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

## RelU will return maximum of 0 and the input dataIn
class RelULayer(Layer):
    def __init__(self, dataIn) -> None:
        super().__init__()
        self.dataIn = dataIn

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.maximum(dataIn, 0)
        self.getPrevOut = dataOut
        return dataOut

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

## I know this one, hard to explain though, but it's the softmax function
class SoftMaxLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        exps = np.exp(dataIn - np.max(dataIn))
        dataOut = exps / np.sum(exps, axis=1, keepdims=True)
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        pass

    def backward(self, gradin):
        pass

## tanH is the hyperbolic tangent function
class tanHLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut =  np.tanh(dataIn)
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        pass

    def backward(self, dataIn):
        pass

class fullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut) -> None:
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.getWeights()
        self.getBias()

     ## the sizeIn x sizeOut matrix is the weight matrix with elements range of ±10^-4
    def getWeights(self):
        self.weights = np.random.uniform(low = -0.0001, high = 0.0001, size = (self.sizeIn, self.sizeOut))

    def setWeights(self, weights):
        self.weights = weights

    ## the sizeOut x 1 matrix is the bias vector with elements range of ±10^-4
    def getBias(self):
        self.bias = np.random.uniform(low = -0.0001, high =  0.0001, size = (1, self.sizeOut))

    def setBias(self, bias):
        self.bias = bias

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.dot(dataIn, self.weights) + self.bias
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass


class fullyConnectedLayer2(Layer):
    ## takes in weights and bias from the user
    def __init__(self, sizeIn, sizeOut, weights, bias) -> None:
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.weights = weights
        self.bias = bias

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.dot(dataIn, self.weights) + self.bias
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass