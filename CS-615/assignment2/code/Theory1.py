# L1 = inputLayer(X)
# L2 = LinearLayer()
# L3 = ReLULayer()
# L4 = SoftmaxLayer()
# L5 = LogisticSigmoid()
# L6 = TanhLayer()
# L7 = fullyConnectedLayer(4,2)
# L8 = LogisticSigmoid()

# layers = [L1, L2, L3, L4, L5, L6, L7, L8]
# Layer = ["input", "Linear", "ReLU", "Softmax", "Logistic", "Tanh", "fullyConnected", "Sigmoid"]

# ## just forward pass and printing the output of each layer along with the layer name
# for i in range(len(layers)):
#     X = layers[i].forward(X)
#     print("\nlayer: ", Layer[i], "\noutput: \n", X, "\n")

# ## getting the gradient of each layer and printing it along with the layer name
# for i in range(len(layers)):
#     grad = layers[i].gradient()
#     print("\nlayer: ", Layer[i], "\ngradient: \n", grad, "\n")

import numpy as np
from layers import *

## Theory 1: Activation functions and their gradients
print("\n-----Activation functions and their gradients---\n")
H = np.array([[1,2,3],[4,5,6]])

## testing the activation functions and their gradients

L1 = ReLULayer() 

print("ReLU: \n", L1.forward(H))
print("\nReLU gradient: \n", L1.gradient())

L2 = SoftmaxLayer()

print("\nSoftmax: \n", L2.forward(H))
print("\nSoftmax gradient: \n", L2.gradient())

L3 = LogisticSigmoid()

print("\nLogistic: \n", L3.forward(H))
print("\nLogistic gradient: \n", L3.gradient())

L4 = TanhLayer()

print("\nTanh: \n", L4.forward(H))
print("\nTanh gradient: \n", L4.gradient())

L5 = LinearLayer()

print("\nLinear: \n", L5.forward(H))
print("\nLinear gradient: \n", L5.gradient())


## Theory 2: Fully connected layer and its gradient

print("\n-----Fully connected layer and its gradient---\n")
W = np.array([[1,2],[3, 4],[5, 6]])
b = np.array([-1, 2])

L6 = fullyConnectedLayer(3,2)

L6.setWeights(W)
L6.setBiases(b)

print("\nfullyConnected: \n", L6.forward(H))
print("\nfullyConnected gradient: \n", L6.gradient())


## Theory 3: Loss functions 
print("\n----Loss functions----\n")

Y = np.array([[0],[1]])
Yhat = np.array([[0.2],[0.3]])

## squared error objective function

from layers import SquaredErrorLoss
L7 = SquaredErrorLoss()

print("\nSquared error: \n", L7.eval(Y, Yhat))

## Negative log likelihood objective function

from layers import NegativeLikelihood

L8 = NegativeLikelihood()

print("\nNegative log likelihood: \n", L8.eval(Y, Yhat))

## Theory 4: Cross entropy loss function

print("\n----Cross entropy loss function----\n")

Y = np.array([[1, 0, 0],[0, 1, 0]])
Yhat = np.array([[0.2, 0.2, 0.6],[0.2, 0.7, 0.1]])

from layers import CrossEntropyLoss

L9 = CrossEntropyLoss()

print("\nCross Entropy Loss: \n", L9.eval(Y, Yhat))

## Theory 5: Objective function gradient 

print("\n----Objective function gradient----\n")

Y = np.array([[0],[1]])
Yhat = np.array([[0.2],[0.3]])

## squared error objective function

print("\nSquared error: \n", L7.gradient(Y, Yhat))

## Negative log likelihood objective function

print("\nNegative log likelihood: \n", L8.gradient(Y, Yhat))


## Theory 6: Cross entropy loss function gradient

Y = np.array([[1, 0, 0],[0, 1, 0]])
Yhat = np.array([[0.2, 0.2, 0.6],[0.2, 0.7, 0.1]])

print("\nCross Entropy Loss: \n", L9.gradient(Y, Yhat))

