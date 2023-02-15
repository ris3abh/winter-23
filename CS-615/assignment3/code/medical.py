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
eva = []
epoch = []
for j in range(10000):
    ## forward pass
    print(j)
    epoch.append(j)
    h = X
    for i in range(len(L)-1):
        h = L[i].forward(h)
    ## evaluating loss
    evaluate = L[-1].eval(Y,h)
    eva.append(evaluate)
    ## backward pass
    grad = L[-1].gradient(Y,h)
    for i in range(len(L)-2,0,-1):
        newgrad = L[i].backward(grad)
        if (isinstance(L[i], fullyConnectedLayer)):
            L[i].update_weights(grad,math.pow(10,-4))
        grad = newgrad
    if j > 1:
        if abs(eva[j] - eva[j-1]) < math.pow(10,-10):
            break

def RMSE(Y, h):
    return np.sqrt(np.mean(np.square(Y - h)))

def SMAPE(Y, h):
    return np.mean(np.abs(Y - h) / (np.abs(Y) + np.abs(h)))

print("RMSE: ", RMSE(Y, h))
print("SMAPE: ", SMAPE(Y, h), "%")

## plotting loss vs epochs
plt.plot(epoch, eva)
plt.xlabel('epochs')
plt.ylabel('squared error loss')
plt.show()

evaluate = np.array(evaluate)
np.save('../data/loss_plot', evaluate)

## SMAPE: 0.1832228728172766 %
## RMSE:  6474.277271413422



