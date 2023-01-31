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
        return dataOut

    def gradient(self):
        pass

    def backward(self, gradIn):
        return gradIn

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut =  dataIn
        self.setPrevOut(dataOut)
        return dataOut

    ## gradient of linear layer is matrices of identity matrix with shape NxKxK
    def gradient(self):
        jacobian = np.zeros((self.getPrevOut().shape[0], self.getPrevOut().shape[1], self.getPrevOut().shape[1]))
        for i in range(self.getPrevOut().shape[0]):
            jacobian[i] = np.identity(self.getPrevOut().shape[1])
        ## since the activation functions gradient were diagonal matrices, we can turn them into single observation’s diagonal in a row, 
        ## then we can reduce the size of the gradient to ∈ ℝ 1×𝐾 for a single observation, and ∈ ℝ 𝑁×𝐾 for multiple observations.
        ## for now lets keep this as a tensor
        return jacobian
        
    def backward(self, gradIn):
        pass

class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.maximum(0, dataIn)
        self.setPrevOut(dataOut)
        return dataOut

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
        pass

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.exp(dataIn - np.max(dataIn))/np.sum(np.exp(dataIn - np.max(dataIn)), axis=1, keepdims=True)
        self.setPrevOut(dataOut)
        return dataOut

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
        # tensor = []
        # for i in prevOut:
        #     tensor.append(np.diag(i) - np.dot(i, i.T))
        # return np.array(tensor)
            

    def backward(self, gradIn):
        pass

class LogisticSigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = 1/(1+np.exp(-dataIn))
        self.setPrevOut(dataOut)
        return dataOut

    
    def gradient(self):
        prevOut = self.getPrevOut()
        jacobian = np.zeros((prevOut.shape[0], prevOut.shape[1], prevOut.shape[1]))
        ## the output of the gradient of sigmoid function will be zero everywhere except the diagonal of the matrix
        ## and will follow the following rule on the diagonal self.getPrevOut() * (1 - self.getPrevOut())
        for i in range(prevOut.shape[0]):
            for j in range(prevOut.shape[1]):
                for k in range(prevOut.shape[1]):
                    if j == k:
                        jacobian[i][j][k] = prevOut[i][j] * (1 - prevOut[i][j])
                    else:
                        jacobian[i][j][k] = 0
        ## since the activation functions gradient were diagonal matrices, we can turn them into single observation’s diagonal in a row, 
        ## then we can reduce the size of the gradient to ∈ ℝ 1×𝐾 for a single observation, and ∈ ℝ 𝑁×𝐾 for multiple observations.
        ## for now lets keep this as a tensor

        return jacobian
        # return self.getPrevOut() * (1 - self.getPrevOut())

    def backward(self, gradIn):
        pass

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.tanh(dataIn)
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        prevOut = self.getPrevOut()
        jacobian = np.zeros((prevOut.shape[0], prevOut.shape[1], prevOut.shape[1]))
        ## the output of the gradient of tanH function will be zero everywhere except the diagonal of the matrix and follow this rule on diagonal  1 - np.square(self.getPrevOut())
        for i in range(prevOut.shape[0]):
            for j in range(prevOut.shape[1]):
                for k in range(prevOut.shape[1]):
                    if j == k:
                        jacobian[i][j][k] = 1 - np.square(self.getPrevOut()[i][j])
                    else:
                        jacobian[i][j][k] = 0
        ## since the activation functions gradient were diagonal matrices, we can turn them into single observation’s diagonal in a row, 
        ## then we can reduce the size of the gradient to ∈ ℝ 1×𝐾 for a single observation, and ∈ ℝ 𝑁×𝐾 for multiple observations.
        ## for now lets keep this as a tensor
        return jacobian
        # return 1 - np.square(self.getPrevOut())

    def backward(self, gradIn):
        pass

class fullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut) -> None:
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.getWeights()
        self.getBias()

    def getWeights(self):
        self.weights = np.random.randn(self.sizeIn, self.sizeOut)

    def setWeights(self, weights):
        self.weights = weights

    def getBias(self):
        self.bias = np.random.randn(self.sizeOut)

    def setBiases(self, biases):
        self.bias = biases

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.dot(dataIn, self.weights) + self.bias
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        prevOut = self.getPrevOut()
        ## jacobian = np.zeros(prevOut[0], prevOut[1], prevOut[1])
        jacobian = []
        for i in range(len(prevOut)):
            jacobian.append(self.weights.T)
        return np.array(jacobian)

        #return np.array([self.weights.T for i in range(len(self.getPrevOut()))])

    def backward(self, gradIn):
        pass

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
        return -np.divide(y, yhat + epsilon) + np.divide((1-y), (1-yhat + epsilon))

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








        


