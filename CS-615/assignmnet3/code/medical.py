import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readCSV():
    dataIn = np.genfromtxt('../data/medical.csv', delimiter=',')
    return dataIn[1:]

X = readCSV()
Y = X[:, [-1]].copy()
X = np.delete(X, -1, axis=1)

from layers import *
import math

L1 = inputLayer(X)
L2 = fullyConnectedLayer(X.shape[1], 1)
L3 = SquaredErrorLoss()

L = [L1, L2, L3]

## for 1000 epochs and learning rate 0.0001 doing forward and backward pass and updating weights and biases for each layer

evaluate = []
h = X
for i in range(10000):
    ## forward pass
    for i in range(len(L)-1):
        h = L[i].forward(h)
    ## evaluating loss
    evaluate.append(L3.eval(Y,h))
    ## backward pass
    grad = L[-1].gradient(Y,h)
    for i in range(len(L)-2,0,-1):
        newgrad = L[i].backward(grad)
        if (isinstance(L[i], fullyConnectedLayer)):
            L[i].update_weights(grad,math.pow(10,-4))
        grad = newgrad

## plotting loss vs epochs
plt.plot(evaluate)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

evaluate = np.array(evaluate)
np.save('../data/loss_plot', evaluate)






