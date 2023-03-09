from abc import ABC, abstractmethod 
import numpy as np
import math

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
    def __init__(self, dataIn, zscore = True):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0, ddof = 1)
        self.stdX[self.stdX == 0] = 1
        self.zscore = True

    def forward(self, dataIn, zscore = True):
        if zscore:
            self.setPrevIn(dataIn)
            dataOut = (dataIn - self.meanX)/self.stdX
            self.setPrevOut(dataOut)
        else:
            self.setPrevIn(dataIn)
            dataOut = dataIn
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
        return np.identity(self.getPrevOut().shape[1])
        
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
        return np.where(prevOut > 0, 1, 0)

    def backward(self, gradIn):
        grad = self.gradient()
        return np.multiply(gradIn, grad)

## non diagonal gradient, so tensor product
class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = self.softmax(dataIn)
        self.setPrevOut(dataOut)
        return self.getPrevOut()
        
    def gradient(self):
        T = []
        for row in self.getPrevOut():
            grad = np.diag(row) - row[np.newaxis].T.dot(row[np.newaxis])
            T.append(grad)
        return np.array(T)

    def backward(self, gradIn):
        grand = self.gradient()
        return np.einsum('ijk,ik->ij', grand, gradIn)

## diagonal gradient, so hadamarts products
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
        grad = self.gradient()
        return np.multiply(gradIn, grad)

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
        return 1 - np.square(prevOut)

    def backward(self, gradIn):
        grad = self.gradient()
        return np.multiply(gradIn, grad)
    
class dropOutLayer(Layer):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.mask = np.random.binomial(1, self.keep_prob, size = dataIn.shape)
        dataOut = np.multiply(dataIn, self.mask)
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        return self.mask

    def backward(self, gradIn):
        grad = self.gradient()
        return np.multiply(gradIn, grad)

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
    

class L2Regularisation(Layer):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda

    def eval(self, error, weights):
        return error + self.lamda * np.sum(np.square(weights))
    
    def gradient(self, error, weights):
        return error + 2 * self.lamda * weights
        
class fullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut) -> None:
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        # self.weights = np.random.uniform(low=-0.0001, high=0.0001, size = (sizeIn, sizeOut))
        # self.baises = np.random.uniform(low=-0.0001, high=0.0001, size = (1, sizeOut))
        ## Xavier initialization
        self.weights = np.random.uniform(low=-np.sqrt(6/(sizeIn + sizeOut)), high=np.sqrt(6/(sizeIn + sizeOut)), size = (sizeIn, sizeOut))
        self.baises = np.random.uniform(low=-np.sqrt(6/(sizeIn + sizeOut)), high=np.sqrt(6/(sizeIn + sizeOut)), size = (1, sizeOut))

        ## for ADAM
        self.biasS, self.biasR = 0, 0
        self.weightS, self.weightR = 0, 0
      
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
        ## prevOut = self.getPrevOut()
        return self.getWeights().T

    def update_weights(self, gradIn, learning_rate, epoch, adam = False, batch_size = 1):
        self.dj_db = np.sum(gradIn, axis = 0)/gradIn.shape[0]
        self.dj_dw = np.dot(self.getPrevIn().T, gradIn)/gradIn.shape[0]
        if adam:
            p1, p2 = 0.9, 0.999
            eta = math.pow(10, -8)
            self.weightS = (p1 * self.weightS) + ((1 - p1) * self.dj_dw)
            self.weightR = (p2 * self.weightR) + ((1 - p2) * (self.dj_dw * self.dj_dw))
            self.biasS = (p1 * self.biasS) + ((1 - p1) * self.dj_db)
            self.biasR = (p2 * self.biasR) + ((1 - p2) * (self.dj_db * self.dj_db))
            self.weights-= learning_rate * (self.weightS/(1 - math.pow(p1, epoch)))/(np.sqrt(self.weightR/(1 - math.pow(p2, epoch))) + eta)
            self.baises-= learning_rate * (self.biasS/(1 - math.pow(p1, epoch)))/(np.sqrt(self.biasR/(1 - math.pow(p2, epoch))) + eta)
        else:
            self.weights-= learning_rate * self.dj_dw
            self.baises-= learning_rate * self.dj_db

    def backward(self, gradIn):
        grad = self.gradient()
        return np.dot(gradIn, grad)