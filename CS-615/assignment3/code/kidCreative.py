import math
import matplotlib.pyplot as plt
import numpy as np
from layers import *


def readCSV():
    dataIn = np.genfromtxt('../data/kidCreative.csv', delimiter=',')
    return dataIn[1:]

X = readCSV()
X = np.delete(X, 0, axis=1) 
Y = X[:, [0]].copy()
X = np.delete(X, 0, axis=1)

L1 = inputLayer(X)
L2 = fullyConnectedLayer(X.shape[1], 1)
L3 = LogisticSigmoid()
L4 = NegativeLikelihood()

L = [L1, L2, L3, L4]

eval = []
epoch = []
for i in range(10000):
    ## forward pass
    print(i)
    epoch.append(i)
    h = X
    for i in range(len(L)-1):
        h = L[i].forward(h)
    ## evaluating loss
    ## h = np.round(h)
    loss = L[-1].eval(Y,h)
    eval.append(loss)
    ## backward pass
    grad = L[-1].gradient(Y,h)
    for i in range(len(L)-2,0,-1):
        newgrad = L[i].backward(grad)
        if (isinstance(L[i], fullyConnectedLayer)):
            L[i].update_weights(grad,math.pow(10,-4))
        grad = newgrad
    if i > 1:
        if math.abs(eval[i] - eval[i-1]) < math.pow(10,-10):
            break

## print("loss:", eval)

def accuracy(Y, h):
    return np.mean(Y == np.round(h))

print("accuracy: ", accuracy(Y, h))

## plotting loss vs epochs
plt.plot(eval)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

## accuracy = 0.8424962852897474


