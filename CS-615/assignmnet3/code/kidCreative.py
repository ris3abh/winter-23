import numpy as np
import matplotlib.pyplot as plt
from layers import *
import math

def readCSV():
    dataIn = np.genfromtxt('../data/kidCreative.csv', delimiter=',')
    return dataIn[1:]

X = readCSV()
X = np.delete(X, 0, axis=1) 
Y = X[:, [0]].copy()
X = np.delete(X, 0, axis=1)

L1 = inputLayer(X)
L2 = fullyConnectedLayer(X.shape[1], 1)
L3 = SoftmaxLayer()
L4 = NegativeLikelihood()

L = [L1, L2, L3, L4]

# def propagation(epoch, learning_rate, h, L, Y):
#     eval = []
#     i = 0
#     while i < 10000:
#         ## forward pass
#         for i in range(len(L)-1):
#             h = L[i].forward(h)
#         h = np.round(h)
#         ## evaluating loss
#         eval.append(L[-1].eval(Y,h))
#         ## backward pass
#         grad = L[-1].gradient(Y,h)
#         for i in range(len(L)-2,0,-1):
#             newgrad = L[i].backward(grad)
#             if (isinstance(L[i], fullyConnectedLayer)):
#                 L[i].update_weights(grad,learning_rate)
#             grad = newgrad
#         if (i > 1):
#             if (abs(eval[i] - eval[i-1]) < math.pow(10,-10)):
#                 break
#         else:
#             continue
#         i += 1
#         print(i)
#         print()
#         print(eval[i])

#     return eval, h

# evaluate, h = propagation(10000, math.pow(10,-4), X, L, Y)

# print("evaluate:", evaluate)
# print("h:", h)

h = X
eval = []
for i in range(len(L)-1):
    h = L[i].forward(h)
h = np.round(h)
## evaluating loss
eval.append(L[-1].eval(Y,h))
## backward pass
grad = L[-1].gradient(Y,h)
# for i in range(len(L)-2,0,-1):
#     newgrad = L[i].backward(grad)
#     if (isinstance(L[i], fullyConnectedLayer)):
#         L[i].update_weights(grad, math.pow(10,-4))
#     grad = newgrad

# print(grad.shape) ## Nx1 shape
# print(L[-2].backward(grad)) ## should be Nx1
## testing the gradient function and backward function for the LogisticSigmoid Layer

L1 = inputLayer(X)
L2 = fullyConnectedLayer(X.shape[1], 1)
L3 = LogisticSigmoid()

L = [L1, L2, L3]

h = X
for i in range(len(L)-1):
    h = L[i].forward(h)
## evaluating loss
eval = NegativeLikelihood().eval(Y,h)

print(h)


