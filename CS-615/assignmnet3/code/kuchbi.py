# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

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

h = X
## forward pass
for i in range(len(L)-1):
    h = L[i].forward(h)
## evaluating loss  
eval = L[-1].eval(Y,h)
## backward pass
grad = L[-1].gradient(Y,h)
newGrad = L[-2].backward(grad)
print(newGrad.shape)