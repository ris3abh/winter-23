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

class inputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0, ddof = 1)
        self.stdX[self.stdX == 0] = 1

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = (dataIn - self.meanX)/self.stdX
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        pass

    def backward(self, gradIn):
        return gradIn*self.gradient()

## diagonal gradient, so hadamarts product
class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut =  dataIn
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        jacobian = np.zeros((self.getPrevOut().shape[0], self.getPrevOut().shape[1], self.getPrevOut().shape[1]))
        for i in range(self.getPrevOut().shape[0]):
            jacobian[i] = np.identity(self.getPrevOut().shape[1])
        return jacobian
        
    def backward(self, gradIn):
        pass

## diagonal gradient, so hadamarts product
class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.maximum(0, dataIn)
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        prevOut = self.getPrevOut()
        jacobian = np.zeros((prevOut.shape[0], prevOut.shape[1], prevOut.shape[1]))
        for i in range(prevOut.shape[0]):
            for j in range(prevOut.shape[1]):
                if prevOut[i][j] > 0:
                    jacobian[i][j][j] = 1
                else:
                    jacobian[i][j][j] = 0
        return jacobian

    def backward(self, gradIn):
        return np.multiply(gradIn, self.gradient())

## non diagonal gradient, so tensor product
class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.exp(dataIn - np.max(dataIn))/np.sum(np.exp(dataIn - np.max(dataIn)), axis=1, keepdims=True)
        self.setPrevOut(dataOut)
        return self.getPrevOut()
        
    def gradient(self):
        prevOut = self.getPrevOut()
        jacobian = np.zeros((prevOut.shape[0], prevOut.shape[1], prevOut.shape[1]))
        for i in range(prevOut.shape[0]):
            for j in range(prevOut.shape[1]):
                for k in range(prevOut.shape[1]):
                    if j == k:
                        jacobian[i][j][k] = prevOut[i][j] * (1 - prevOut[i][j])
                    else:
                        jacobian[i][j][k] = -prevOut[i][j] * prevOut[i][k]
        return jacobian
            

    def backward(self, gradIn):
        return np.tensordot(gradIn, self.gradient(), axes=0)

## diagonal gradient, so hadamarts product
class LogisticSigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = 1/(1+np.exp(-dataIn))
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        prevOut = self.getPrevOut()
        return self.getPrevOut() * (1 - self.getPrevOut())

    def backward(self, gradIn):
        return np.multiply(gradIn, self.gradient())

## diagonal gradient, so hadamarts product
class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.tanh(dataIn)
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        prevOut = self.getPrevOut()
        jacobian = np.zeros((prevOut.shape[0], prevOut.shape[1], prevOut.shape[1]))
        for i in range(prevOut.shape[0]):
            for j in range(prevOut.shape[1]):
                for k in range(prevOut.shape[1]):
                    if j == k:
                        jacobian[i][j][k] = 1 - np.square(self.getPrevOut()[i][j])
                    else:
                        jacobian[i][j][k] = 0
        return jacobian
        # return 1 - np.square(self.getPrevOut())

    def backward(self, gradIn):
        return np.multiply(gradIn, self.gradient())

## for the fully connected layer, matrix multiplication is used to calculate the output of the backpropagation
class fullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut) -> None:
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.weights = np.random.uniform(low=-0.0001, high=0.0001, size = (sizeIn, sizeOut))
        self.baises = np.random.uniform(low=-0.0001, high=0.0001, size = (1, sizeOut))

    def getWeights(self):
        return self.weights

    def getBaises(self):
        return self.baises

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.dot(dataIn, self.weights) + self.baises
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        prevOut = self.getPrevOut()
        jacobian = []
        for i in range(len(prevOut)):
            jacobian.append(self.weights.T)
        return np.array(jacobian)

    def update_weights(self, gradIn, lr):
        self.dj_db = np.sum(gradIn, axis = 0)/gradIn.shape[0]
        self.dj_dw = np.dot(self.getPrevIn().T, gradIn)/gradIn.shape[0]
        self.weights = self.weights - (lr*self.dj_dw)
        self.baises = self.baises - (lr*self.dj_db)

    def backward(self, gradIn):
        return np.dot(gradIn, self.gradient())

class SquaredErrorLoss():
    def __init__(self):
        super().__init__()
        self.y = None
        self.yhat = None

    def eval(self, y, yhat):
        self.y = y
        self.yhat = yhat
        return np.mean((y - yhat)*(y - yhat))

    def gradient(self, y, yhat):
        return 2*(yhat - y)

class NegativeLikelihood():
    def __init__(self):
        super().__init__()
        self.y = None
        self.yhat = None

    def eval(self, y , yhat):
        epsilon = 0.0000001
        self.y = y
        self.yhat = yhat
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        ## should return a single value

    def gradient(self, y, yhat):
        epsilon = 0.0000001
        return - (y - yhat)/ (yhat * (1 - yhat) + epsilon)

class CrossEntropyLoss():
    def __init__(self):
        super().__init__()
        self.y = None
        self.yhat = None

    def eval(self, y, yhat):
        epsilon = 0.0000001
        self.y = y
        self.yhat = yhat
        return -np.mean(np.sum(np.multiply(y, np.log(yhat + epsilon)), axis=1))

    def gradient(self, y, yhat):
        epsilon = 0.0000001
        return -np.divide(y, yhat + epsilon)


## end of layers